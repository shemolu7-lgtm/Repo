#!/usr/bin/env python3
"""Single-file Python chess engine with interactive and UCI modes.

Requires: python-chess and numpy. Optional: a Polyglot opening book named
book.bin; the engine searches for it below the current directory, home, and
common share paths.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.polyglot
import numpy as np

INF = 10_000_000
MATE = 9_000_000
MAX_PLY = 128
TT_SIZE = 1 << 20

PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000}
VICTIM = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20}
PHASE_WEIGHT = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}

# PeSTO-like middle-game tables from White's perspective; endgame values are
# tapered from them to keep this compact and single-file.
MG_PST = {
    chess.PAWN: [0,0,0,0,0,0,0,0, 98,134,61,95,68,126,34,-11, -6,7,26,31,65,56,25,-20, -14,13,6,21,23,12,17,-23, -27,-2,-5,12,17,6,10,-25, -26,-4,-4,-10,3,3,33,-12, -35,-1,-20,-23,-15,24,38,-22, 0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-167,-89,-34,-49,61,-97,-15,-107, -73,-41,72,36,23,62,7,-17, -47,60,37,65,84,129,73,44, -9,17,19,53,37,69,18,22, -13,4,16,13,28,19,21,-8, -23,-9,12,10,19,17,25,-16, -29,-53,-12,-3,-1,18,-14,-19, -105,-21,-58,-33,-17,-28,-19,-23],
    chess.BISHOP: [-29,4,-82,-37,-25,-42,7,-8, -26,16,-18,-13,30,59,18,-47, -16,37,43,40,35,50,37,-2, -4,5,19,50,37,37,7,-2, -6,13,13,26,34,12,10,4, 0,15,15,15,14,27,18,10, 4,15,16,0,7,21,33,1, -33,-3,-14,-21,-13,-12,-39,-21],
    chess.ROOK: [32,42,32,51,63,9,31,43, 27,32,58,62,80,67,26,44, -5,19,26,36,17,45,61,16, -24,-11,7,26,24,35,-8,-20, -36,-26,-12,-1,9,-7,6,-23, -45,-25,-16,-17,3,0,-5,-33, -44,-16,-20,-9,-1,11,-6,-71, -19,-13,1,17,16,7,-37,-26],
    chess.QUEEN: [-28,0,29,12,59,44,43,45, -24,-39,-5,1,-16,57,28,54, -13,-17,7,8,29,56,47,57, -27,-27,-16,-16,-1,17,-2,1, -9,-26,-9,-10,-2,-4,3,-3, -14,2,-11,-2,-5,2,14,5, -35,-8,11,2,8,15,-3,1, -1,-18,-9,10,-15,-25,-31,-50],
    chess.KING: [-65,23,16,-15,-56,-34,2,13, 29,-1,-20,-7,-8,-4,-38,-29, -9,24,2,-16,-20,6,22,-22, -17,-20,-12,-27,-30,-25,-14,-36, -49,-1,-27,-39,-46,-44,-33,-51, -14,-14,-22,-46,-44,-30,-15,-27, 1,7,-8,-64,-43,-16,9,8, -15,36,12,-54,8,-28,24,14],
}
EG_PST = {p: [int(v * 0.55) for v in vals] for p, vals in MG_PST.items()}

@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int  # 0 exact, -1 upper, 1 lower
    move: Optional[chess.Move]
    age: int

@dataclass
class SearchInfo:
    nodes: int = 0
    qnodes: int = 0
    depth: int = 0
    seldepth: int = 0
    start: float = 0.0
    soft_limit: float = 1.0
    hard_limit: float = 1.2
    stop: bool = False
    best: Optional[chess.Move] = None
    score: int = 0  # always White perspective at root
    pv: List[chess.Move] = field(default_factory=list)
    max_nodes: Optional[int] = None

class BoundedTT:
    """Fixed-size depth/age-aware TT to avoid unbounded dict growth."""
    def __init__(self, size: int = TT_SIZE) -> None:
        self.size = size
        self.table: List[Optional[TTEntry]] = [None] * size
        self.age = 0

    def new_search(self) -> None:
        self.age = (self.age + 1) & 255

    def clear(self) -> None:
        self.table = [None] * self.size
        self.age = 0

    def get(self, key: int) -> Optional[TTEntry]:
        e = self.table[key % self.size]
        return e if e and e.key == key else None

    def put(self, key: int, depth: int, score: int, flag: int, move: Optional[chess.Move]) -> None:
        idx = key % self.size
        old = self.table[idx]
        if old is None or old.key != key or depth >= old.depth - 1 or old.age != self.age or flag == 0:
            self.table[idx] = TTEntry(key, depth, score, flag, move, self.age)

class Engine:
    def __init__(self) -> None:
        self.tt = BoundedTT()
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.capture_history = np.zeros((2, 64, 64), dtype=np.int32)
        self.counter = [[None for _ in range(64)] for _ in range(64)]
        self.continuation = np.zeros((64, 64, 64, 64), dtype=np.int16)
        self.pawn_cache: Dict[Tuple[int, int], int] = {}
        self.book = self.find_book()

    def find_book(self) -> Optional[Path]:
        for root in (Path.cwd(), Path.home(), Path('/usr/share'), Path('/usr/local/share')):
            if not root.exists():
                continue
            direct = root / 'book.bin'
            if direct.exists():
                return direct
            try:
                found = next(root.rglob('book.bin'), None)
                if found:
                    return found
            except (OSError, PermissionError):
                pass
        return None

    def key(self, board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board)

    def non_pawn_material(self, board: chess.Board, color: chess.Color) -> int:
        return sum(PIECE_VALUE[p] * len(board.pieces(p, color)) for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))

    def evaluate_white(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE if board.turn == chess.WHITE else MATE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        mg = eg = phase = 0
        for sq, pc in board.piece_map().items():
            sign = 1 if pc.color == chess.WHITE else -1
            idx = sq if pc.color == chess.WHITE else chess.square_mirror(sq)
            val = PIECE_VALUE[pc.piece_type]
            if pc.piece_type == chess.KING:
                val = 0
            mg += sign * (val + MG_PST[pc.piece_type][idx])
            eg += sign * (val + EG_PST[pc.piece_type][idx])
            phase += PHASE_WEIGHT[pc.piece_type]
        phase = min(24, phase)
        score = (mg * phase + eg * (24 - phase)) // 24
        score += self.pawn_structure(board)
        score += self.king_safety(board)
        score += self.mobility(board)
        score += self.positional_terms(board)
        return int(self.scale_endgame(board, score))

    def evaluate_stm(self, board: chess.Board) -> int:
        s = self.evaluate_white(board)
        return s if board.turn == chess.WHITE else -s

    def pawn_structure(self, board: chess.Board) -> int:
        key = (int(board.pawns & board.occupied_co[chess.WHITE]), int(board.pawns & board.occupied_co[chess.BLACK]))
        if key in self.pawn_cache:
            return self.pawn_cache[key]
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            pawns = board.pieces(chess.PAWN, color)
            enemy = board.pieces(chess.PAWN, not color)
            files = [0] * 8
            for sq in pawns:
                files[chess.square_file(sq)] += 1
            score -= sign * 12 * sum(max(0, n - 1) for n in files)
            for sq in pawns:
                f, r = chess.square_file(sq), chess.square_rank(sq)
                if not any(files[x] for x in range(max(0, f - 1), min(7, f + 1) + 1) if x != f):
                    score -= sign * 10
                ahead = range(r + 1, 8) if color == chess.WHITE else range(r - 1, -1, -1)
                passed = not any(chess.square(ff, rr) in enemy for ff in range(max(0, f - 1), min(7, f + 1) + 1) for rr in ahead)
                if passed:
                    rel = r if color == chess.WHITE else 7 - r
                    block = chess.square(f, r + (1 if color == chess.WHITE else -1)) if 0 <= r + (1 if color == chess.WHITE else -1) < 8 else sq
                    bonus = 10 + rel * rel * 3
                    if board.piece_at(block):
                        bonus -= 10
                    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
                    if wk is not None and bk is not None:
                        promo = chess.square(f, 7 if color == chess.WHITE else 0)
                        friendly = wk if color == chess.WHITE else bk
                        enemyk = bk if color == chess.WHITE else wk
                        bonus += chess.square_distance(enemyk, promo) * 3 - chess.square_distance(friendly, promo) * 2
                    score += sign * bonus
        self.pawn_cache[key] = score
        return score

    def king_safety(self, board: chess.Board) -> int:
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            k = board.king(color)
            if k is None:
                continue
            shield = 0
            direction = 1 if color == chess.WHITE else -1
            for df in (-1, 0, 1):
                f, r = chess.square_file(k) + df, chess.square_rank(k) + direction
                if 0 <= f < 8 and 0 <= r < 8 and board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color):
                    shield += 1
            zone = set(board.attacks(k)) | {k}
            attack_units = 0
            for sq in zone:
                attack_units += 2 * len(board.attackers(not color, sq))
                attack_units -= len(board.attackers(color, sq))
            queen_attack = any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.QUEEN for sq in board.pieces(chess.QUEEN, not color))
            score += sign * (14 * shield - 9 * max(0, attack_units) - (8 if queen_attack and attack_units > 3 else 0))
        return score

    def mobility(self, board: chess.Board) -> int:
        turn = board.turn
        board.turn = chess.WHITE; wm = board.legal_moves.count()
        board.turn = chess.BLACK; bm = board.legal_moves.count()
        board.turn = turn
        return 2 * (wm - bm)

    def positional_terms(self, board: chess.Board) -> int:
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            own_pawns = board.pieces(chess.PAWN, color)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                score += sign * 35
            for rook in board.pieces(chess.ROOK, color):
                f = chess.square_file(rook)
                own = any(chess.square(f, rr) in own_pawns for rr in range(8))
                enemy = any(chess.square(f, rr) in enemy_pawns for rr in range(8))
                score += sign * (18 if not own and not enemy else 10 if not own else 0)
            for knight in board.pieces(chess.KNIGHT, color):
                r = chess.square_rank(knight) if color == chess.WHITE else 7 - chess.square_rank(knight)
                protected = bool(board.attackers(color, knight) & own_pawns)
                enemy_pawn_attack = bool(board.attackers(not color, knight) & enemy_pawns)
                if r >= 3 and protected and not enemy_pawn_attack:
                    score += sign * 22
            for pt, weight in [(chess.KNIGHT, 4), (chess.BISHOP, 4), (chess.ROOK, 2), (chess.QUEEN, 1)]:
                for sq in board.pieces(pt, color):
                    score += sign * weight * len(board.attacks(sq))
        return score

    def scale_endgame(self, board: chess.Board, score: int) -> int:
        total = self.non_pawn_material(board, chess.WHITE) + self.non_pawn_material(board, chess.BLACK)
        if total <= 700:
            # Opposite-colored bishops and very low material are drawish.
            wb, bb = board.pieces(chess.BISHOP, chess.WHITE), board.pieces(chess.BISHOP, chess.BLACK)
            if len(wb) == len(bb) == 1:
                if chess.square_color(next(iter(wb))) != chess.square_color(next(iter(bb))):
                    score = int(score * 0.65)
            if not board.pieces(chess.PAWN, chess.WHITE) and not board.pieces(chess.PAWN, chess.BLACK):
                score = int(score * 0.4)
        return score

    def see(self, board: chess.Board, move: chess.Move) -> int:
        """Swap-list static exchange evaluation on the target square."""
        if not board.is_capture(move) and not move.promotion:
            return 0
        b = board.copy(stack=False)
        target = move.to_square
        victim = b.piece_at(target)
        gain = [PIECE_VALUE.get(victim.piece_type, 0) if victim else 0]
        if move.promotion:
            gain[0] += PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
        side = b.turn
        b.push(move)
        side = not side
        depth = 0
        while True:
            attackers = list(b.attackers(side, target))
            attackers = [sq for sq in attackers if b.piece_at(sq) and b.piece_at(sq).color == side]
            if not attackers:
                break
            frm = min(attackers, key=lambda sq: PIECE_VALUE[b.piece_at(sq).piece_type])
            pc = b.piece_at(frm)
            gain.append(PIECE_VALUE[pc.piece_type] - gain[depth])
            if max(-gain[depth], gain[depth + 1]) < 0:
                break
            b.remove_piece_at(frm)
            b.set_piece_at(target, pc)
            side = not side
            depth += 1
        while depth > 0:
            depth -= 1
            gain[depth] = -max(-gain[depth], gain[depth + 1])
        return gain[0]

    def move_score(self, board: chess.Board, move: chess.Move, ply: int, ttmove: Optional[chess.Move], prev: Optional[chess.Move]) -> int:
        if move == ttmove: return 2_000_000
        if prev and self.counter[prev.from_square][prev.to_square] == move: return 950_000
        if move.promotion: return 800_000 + PIECE_VALUE.get(move.promotion, 0)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square) or chess.Piece(chess.PAWN, not board.turn)
            attacker = board.piece_at(move.from_square)
            return 600_000 + 100 * VICTIM[victim.piece_type] - VICTIM.get(attacker.piece_type, 0) + self.see(board, move) + int(self.capture_history[int(board.turn), move.from_square, move.to_square])
        if board.gives_check(move): return 500_000
        if move in self.killers[ply]: return 400_000
        cont = int(self.continuation[prev.from_square, prev.to_square, move.from_square, move.to_square]) if prev else 0
        return int(self.history[int(board.turn), move.from_square, move.to_square]) + cont

    def ordered_moves(self, board: chess.Board, ply: int, ttmove: Optional[chess.Move], prev: Optional[chess.Move]) -> List[chess.Move]:
        return sorted(board.legal_moves, key=lambda m: self.move_score(board, m, ply, ttmove, prev), reverse=True)

    def should_stop(self, info: SearchInfo) -> bool:
        if info.max_nodes is not None and info.nodes + info.qnodes >= info.max_nodes:
            info.stop = True
        if time.time() - info.start >= info.hard_limit:
            info.stop = True
        return info.stop

    def search_root(self, board: chess.Board, seconds: float, max_nodes: Optional[int] = None, max_depth: int = 64) -> SearchInfo:
        self.tt.new_search()
        info = SearchInfo(start=time.time(), soft_limit=max(0.03, seconds), hard_limit=max(0.05, seconds * 1.35), max_nodes=max_nodes)
        book_move = self.book_move(board)
        if book_move:
            info.best = book_move; info.score = self.evaluate_white(board); info.pv = [book_move]
            return info
        last, stable, prev_best = 0, 0, None
        for depth in range(1, max_depth + 1):
            if time.time() - info.start >= info.soft_limit and depth > 1:
                break
            window = 18 if depth >= 5 else INF
            alpha, beta = last - window, last + window
            while True:
                score, pv = self.pvs(board, depth, alpha, beta, 0, info, None, True, True)
                if info.stop:
                    break
                if score <= alpha:
                    alpha -= window; window *= 2
                elif score >= beta:
                    beta += window; window *= 2
                else:
                    last = score
                    info.best = pv[0] if pv else info.best
                    info.pv = pv
                    info.score = score if board.turn == chess.WHITE else -score
                    info.depth = depth
                    stable = stable + 1 if info.best == prev_best else 0
                    prev_best = info.best
                    break
            if info.stop:
                break
            if depth >= 4 and stable >= 2 and time.time() - info.start >= info.soft_limit * 0.65:
                break
        return info

    def pvs(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, info: SearchInfo, prev: Optional[chess.Move], null_ok: bool = True, root: bool = False) -> Tuple[int, List[chess.Move]]:
        if (info.nodes & 2047) == 0 and self.should_stop(info):
            return 0, []
        info.nodes += 1; info.seldepth = max(info.seldepth, ply)
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha, []
        if board.is_checkmate(): return -MATE + ply, []
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw(): return 0, []
        in_check = board.is_check()
        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply, info), []

        key = self.key(board); entry = self.tt.get(key); ttmove = entry.move if entry else None
        if entry and entry.depth >= depth and not root:
            if entry.flag == 0: return entry.score, [entry.move] if entry.move else []
            if entry.flag == 1 and entry.score >= beta: return entry.score, [entry.move] if entry.move else []
            if entry.flag == -1 and entry.score <= alpha: return entry.score, [entry.move] if entry.move else []

        # Internal iterative deepening supplies a likely TT/PV move at deep nodes.
        if ttmove is None and depth >= 4 and not in_check:
            _, iid_pv = self.pvs(board, depth - 2, alpha, beta, ply, info, prev, False)
            if iid_pv:
                ttmove = iid_pv[0]

        static = self.evaluate_stm(board)
        if not in_check and depth <= 3 and static - 90 * depth >= beta:
            return static, []
        if not in_check and depth <= 2 and static + 120 * depth <= alpha:
            return alpha, []
        if null_ok and not in_check and depth >= 3 and self.non_pawn_material(board, board.turn) > 500 and abs(static) < MATE // 2:
            reduction = 2 + depth // 5 + int(static - beta > 150)
            board.push(chess.Move.null())
            score, _ = self.pvs(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, info, None, False)
            board.pop()
            if info.stop: return 0, []
            if -score >= beta:
                if depth >= 7:
                    verify, _ = self.pvs(board, depth - reduction, beta - 1, beta, ply, info, prev, False)
                    if verify >= beta: return beta, []
                else:
                    return beta, []

        best, best_pv, best_score, old_alpha = None, [], -INF, alpha
        moves = self.ordered_moves(board, ply, ttmove, prev)
        for i, move in enumerate(moves):
            quiet = not board.is_capture(move) and not board.gives_check(move) and not move.promotion
            if quiet and depth <= 2 and not in_check and static + 100 * depth <= alpha:
                continue
            ext = 0
            if in_check or move.promotion:
                ext = 1
            elif prev and move.to_square == prev.to_square and board.is_capture(move):
                ext = 1
            elif ttmove == move and depth >= 7 and entry and entry.depth >= depth - 2 and entry.score >= beta - 60:
                ext = 1
            reduction = 0
            if quiet and depth >= 3 and i >= 4 and not in_check and ext == 0:
                reduction = 1 + int(depth >= 5 and i >= 8)
            board.push(move)
            new_depth = depth - 1 + ext
            if i == 0:
                score, child = self.pvs(board, new_depth, -beta, -alpha, ply + 1, info, move, True)
                score = -score
            else:
                score, child = self.pvs(board, max(0, new_depth - reduction), -alpha - 1, -alpha, ply + 1, info, move, True)
                score = -score
                if score > alpha and reduction:
                    score, child = self.pvs(board, new_depth, -alpha - 1, -alpha, ply + 1, info, move, True); score = -score
                if alpha < score < beta:
                    score, child = self.pvs(board, new_depth, -beta, -alpha, ply + 1, info, move, True); score = -score
            board.pop()
            if info.stop: return 0, []
            if score > best_score:
                best_score, best, best_pv = score, move, [move] + child
            if score > alpha:
                alpha = score
            if alpha >= beta:
                side = int(board.turn)
                if quiet:
                    self.killers[ply][1] = self.killers[ply][0]; self.killers[ply][0] = move
                    self.history[side, move.from_square, move.to_square] += depth * depth
                    if prev:
                        self.counter[prev.from_square][prev.to_square] = move
                        self.continuation[prev.from_square, prev.to_square, move.from_square, move.to_square] += min(1000, depth * depth)
                else:
                    self.capture_history[side, move.from_square, move.to_square] += depth * depth
                break
        if best is None:
            return self.qsearch(board, alpha, beta, ply, info), []
        flag = 0 if best_score > old_alpha and best_score < beta else (1 if best_score >= beta else -1)
        self.tt.put(key, depth, best_score, flag, best)
        return best_score, best_pv

    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, info: SearchInfo) -> int:
        if self.should_stop(info):
            return 0
        info.qnodes += 1; info.seldepth = max(info.seldepth, ply)
        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
        else:
            stand = self.evaluate_stm(board)
            if stand >= beta: return beta
            alpha = max(alpha, stand)
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion or board.gives_check(m)]
        moves.sort(key=lambda m: self.move_score(board, m, min(ply, MAX_PLY - 1), None, None), reverse=True)
        for move in moves:
            if not in_check and board.is_capture(move) and self.see(board, move) < -60:
                continue
            board.push(move)
            score = -self.qsearch(board, -beta, -alpha, ply + 1, info)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    def book_move(self, board: chess.Board) -> Optional[chess.Move]:
        if not self.book: return None
        try:
            with chess.polyglot.open_reader(str(self.book)) as reader:
                return reader.weighted_choice(board).move
        except Exception:
            return None

    def go(self, board: chess.Board, movetime: float, nodes: Optional[int] = None, depth: int = 64) -> SearchInfo:
        return self.search_root(board, max(0.03, movetime), nodes, depth)

def fmt_eval(cp: int) -> str:
    if abs(cp) > MATE - 1000:
        return f"mate {int(math.copysign((MATE - abs(cp) + 1) // 2, cp))}"
    return f"{cp/100:.2f}"

def parse_user_move(board: chess.Board, text: str) -> Optional[chess.Move]:
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError:
        return None

def interactive() -> None:
    engine, board = Engine(), chess.Board()
    color = input("Play as white or black? [w/b]: ").strip().lower().startswith('w')
    tpm = float(input("Time per engine move in seconds: ").strip() or "2")
    print(board)
    while not board.is_game_over():
        if board.turn == color:
            move = parse_user_move(board, input("Your move (SAN or UCI): ").strip())
            if move is None or move not in board.legal_moves:
                print("Illegal or invalid move."); continue
            board.push(move)
        else:
            info = engine.go(board, tpm)
            move = info.best or next(iter(board.legal_moves))
            board.push(move)
            elapsed = max(1e-6, time.time() - info.start)
            nodes = info.nodes + info.qnodes
            pv = ' '.join(m.uci() for m in info.pv)
            print(f"Engine move: {move.uci()}")
            print(f"nodes={nodes} depth={info.depth} seldepth={info.seldepth} nps={int(nodes/elapsed)} time={elapsed:.3f}s eval_white={fmt_eval(info.score)} pv={pv}")
        print(board, "\nFEN:", board.fen())
    print("Game over:", board.result(), board.outcome())

def time_from_go(parts: List[str], board: chess.Board) -> Tuple[float, Optional[int], int]:
    depth, nodes = 64, None
    if 'depth' in parts: depth = int(parts[parts.index('depth') + 1])
    if 'nodes' in parts: nodes = int(parts[parts.index('nodes') + 1])
    if 'movetime' in parts: return max(0.03, int(parts[parts.index('movetime') + 1]) / 1000), nodes, depth
    if 'infinite' in parts: return 24 * 3600, nodes, depth
    side_time = 'wtime' if board.turn == chess.WHITE else 'btime'
    side_inc = 'winc' if board.turn == chess.WHITE else 'binc'
    if side_time in parts:
        remaining = int(parts[parts.index(side_time) + 1]) / 1000
        inc = int(parts[parts.index(side_inc) + 1]) / 1000 if side_inc in parts else 0
        mtg = int(parts[parts.index('movestogo') + 1]) if 'movestogo' in parts else 30
        return max(0.03, min(remaining * 0.25, remaining / max(1, mtg) + 0.75 * inc - 0.03)), nodes, depth
    return 1.0, nodes, depth

def uci() -> None:
    engine, board = Engine(), chess.Board()
    while True:
        line = sys.stdin.readline()
        if not line: break
        parts = line.strip().split()
        if not parts: continue
        cmd = parts[0]
        if cmd == 'uci':
            print('id name StrongSingleFilePyEngine'); print('id author OpenAI'); print('uciok')
        elif cmd == 'isready': print('readyok')
        elif cmd == 'ucinewgame': board.reset(); engine.tt.clear(); engine.pawn_cache.clear()
        elif cmd == 'position':
            idx = 1
            if idx < len(parts) and parts[idx] == 'startpos': board = chess.Board(); idx += 1
            elif idx < len(parts) and parts[idx] == 'fen':
                board = chess.Board(' '.join(parts[idx+1:idx+7])); idx += 7
            if idx < len(parts) and parts[idx] == 'moves':
                for m in parts[idx+1:]: board.push(chess.Move.from_uci(m))
        elif cmd == 'go':
            mt, nodes, depth = time_from_go(parts, board)
            info = engine.go(board, mt, nodes, depth)
            total = info.nodes + info.qnodes
            uci_score = info.score if board.turn == chess.WHITE else -info.score
            pv = ' '.join(m.uci() for m in info.pv)
            print(f"info depth {info.depth} seldepth {info.seldepth} score cp {uci_score} nodes {total} nps {int(total/max(1e-6,time.time()-info.start))} pv {pv}")
            print('bestmove', (info.best or next(iter(board.legal_moves))).uci())
        elif cmd == 'quit': break
        sys.stdout.flush()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'uci':
        uci()
    else:
        interactive()
