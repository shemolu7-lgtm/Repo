#!/usr/bin/env python3
"""A single-file Python chess engine with interactive and UCI modes.

Requires: python-chess and numpy. Optional: a Polyglot book named book.bin
somewhere below the current directory, home directory, or common book paths.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.polyglot
import numpy as np

INF = 10_000_000
MATE = 9_000_000
DRAW = 0
MAX_PLY = 128

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
VICTIM = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 20}

# Simplified PeSTO-like middle-game/end-game piece-square tables, from White's view.
MG_PST = {
    chess.PAWN: [0,0,0,0,0,0,0,0, 98,134,61,95,68,126,34,-11, -6,7,26,31,65,56,25,-20, -14,13,6,21,23,12,17,-23, -27,-2,-5,12,17,6,10,-25, -26,-4,-4,-10,3,3,33,-12, -35,-1,-20,-23,-15,24,38,-22, 0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-167,-89,-34,-49,61,-97,-15,-107, -73,-41,72,36,23,62,7,-17, -47,60,37,65,84,129,73,44, -9,17,19,53,37,69,18,22, -13,4,16,13,28,19,21,-8, -23,-9,12,10,19,17,25,-16, -29,-53,-12,-3,-1,18,-14,-19, -105,-21,-58,-33,-17,-28,-19,-23],
    chess.BISHOP: [-29,4,-82,-37,-25,-42,7,-8, -26,16,-18,-13,30,59,18,-47, -16,37,43,40,35,50,37,-2, -4,5,19,50,37,37,7,-2, -6,13,13,26,34,12,10,4, 0,15,15,15,14,27,18,10, 4,15,16,0,7,21,33,1, -33,-3,-14,-21,-13,-12,-39,-21],
    chess.ROOK: [32,42,32,51,63,9,31,43, 27,32,58,62,80,67,26,44, -5,19,26,36,17,45,61,16, -24,-11,7,26,24,35,-8,-20, -36,-26,-12,-1,9,-7,6,-23, -45,-25,-16,-17,3,0,-5,-33, -44,-16,-20,-9,-1,11,-6,-71, -19,-13,1,17,16,7,-37,-26],
    chess.QUEEN: [-28,0,29,12,59,44,43,45, -24,-39,-5,1,-16,57,28,54, -13,-17,7,8,29,56,47,57, -27,-27,-16,-16,-1,17,-2,1, -9,-26,-9,-10,-2,-4,3,-3, -14,2,-11,-2,-5,2,14,5, -35,-8,11,2,8,15,-3,1, -1,-18,-9,10,-15,-25,-31,-50],
    chess.KING: [-65,23,16,-15,-56,-34,2,13, 29,-1,-20,-7,-8,-4,-38,-29, -9,24,2,-16,-20,6,22,-22, -17,-20,-12,-27,-30,-25,-14,-36, -49,-1,-27,-39,-46,-44,-33,-51, -14,-14,-22,-46,-44,-30,-15,-27, 1,7,-8,-64,-43,-16,9,8, -15,36,12,-54,8,-28,24,14],
}
EG_PST = {p: [int(v * 0.55) for v in vals] for p, vals in MG_PST.items()}
PHASE_WEIGHT = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}

@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int  # 0 exact, -1 upper, 1 lower
    move: Optional[chess.Move]

@dataclass
class SearchInfo:
    nodes: int = 0
    qnodes: int = 0
    depth: int = 0
    seldepth: int = 0
    start: float = 0.0
    limit: float = 1.0
    stop: bool = False
    best: Optional[chess.Move] = None
    score: int = 0

class Engine:
    def __init__(self) -> None:
        self.tt: Dict[int, TTEntry] = {}
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.book = self.find_book()

    def find_book(self) -> Optional[Path]:
        names = ["book.bin"]
        roots = [Path.cwd(), Path.home(), Path('/usr/share'), Path('/usr/local/share')]
        for root in roots:
            if not root.exists():
                continue
            for name in names:
                direct = root / name
                if direct.exists():
                    return direct
            try:
                for p in root.rglob('book.bin'):
                    return p
            except (OSError, PermissionError):
                pass
        return None

    def key(self, board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board)

    def evaluate_white(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE if board.turn == chess.WHITE else MATE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return DRAW
        mg = eg = phase = 0
        for sq, pc in board.piece_map().items():
            sign = 1 if pc.color == chess.WHITE else -1
            idx = sq if pc.color == chess.WHITE else chess.square_mirror(sq)
            mg += sign * (PIECE_VALUE[pc.piece_type] + MG_PST[pc.piece_type][idx])
            eg += sign * (PIECE_VALUE[pc.piece_type] + EG_PST[pc.piece_type][idx])
            phase += PHASE_WEIGHT[pc.piece_type]
        phase = min(24, phase)
        score = (mg * phase + eg * (24 - phase)) // 24
        score += self.pawn_structure(board)
        score += self.king_safety(board)
        score += self.mobility(board)
        score += self.positional_terms(board)
        return int(score)

    def evaluate_stm(self, board: chess.Board) -> int:
        score = self.evaluate_white(board)
        return score if board.turn == chess.WHITE else -score

    def pawn_structure(self, board: chess.Board) -> int:
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            pawns = board.pieces(chess.PAWN, color)
            files = [0] * 8
            for sq in pawns:
                files[chess.square_file(sq)] += 1
            score -= sign * 12 * sum(max(0, n - 1) for n in files)
            for sq in pawns:
                f = chess.square_file(sq)
                if not any(files[x] for x in range(max(0, f - 1), min(7, f + 1) + 1) if x != f):
                    score -= sign * 10
        return score

    def king_safety(self, board: chess.Board) -> int:
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            k = board.king(color)
            if k is None:
                continue
            shield = 0
            rank_dir = 1 if color == chess.WHITE else -1
            for df in (-1, 0, 1):
                f = chess.square_file(k) + df
                r = chess.square_rank(k) + rank_dir
                if 0 <= f < 8 and 0 <= r < 8 and board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color):
                    shield += 1
            attackers = len(board.attackers(not color, k))
            score += sign * (12 * shield - 18 * attackers)
        return score

    def mobility(self, board: chess.Board) -> int:
        turn = board.turn
        board.turn = chess.WHITE; wm = board.legal_moves.count()
        board.turn = chess.BLACK; bm = board.legal_moves.count()
        board.turn = turn
        return 2 * (wm - bm)


    def positional_terms(self, board: chess.Board) -> int:
        """Extra Sunfish-strength positional features, returned from White's view."""
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            bishops = len(board.pieces(chess.BISHOP, color))
            if bishops >= 2:
                score += sign * 35
            enemy_pawns = board.pieces(chess.PAWN, not color)
            own_pawns = board.pieces(chess.PAWN, color)
            for sq in own_pawns:
                f, r = chess.square_file(sq), chess.square_rank(sq)
                ahead = range(r + 1, 8) if color == chess.WHITE else range(r - 1, -1, -1)
                if not any(chess.square(ff, rr) in enemy_pawns for ff in range(max(0, f - 1), min(7, f + 1) + 1) for rr in ahead):
                    bonus_rank = r if color == chess.WHITE else 7 - r
                    score += sign * (8 + bonus_rank * bonus_rank)
            for rook in board.pieces(chess.ROOK, color):
                f = chess.square_file(rook)
                own_on_file = any(chess.square(f, rr) in own_pawns for rr in range(8))
                enemy_on_file = any(chess.square(f, rr) in enemy_pawns for rr in range(8))
                if not own_on_file and not enemy_on_file:
                    score += sign * 18
                elif not own_on_file:
                    score += sign * 10
            for pt, weight in [(chess.KNIGHT, 4), (chess.BISHOP, 4), (chess.ROOK, 2), (chess.QUEEN, 1)]:
                for sq in board.pieces(pt, color):
                    score += sign * weight * len(board.attacks(sq))
        return score

    def see(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if move.promotion:
            promo_gain = PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
        else:
            promo_gain = 0
        return (PIECE_VALUE.get(victim.piece_type, 0) if victim else 0) + promo_gain - (PIECE_VALUE.get(attacker.piece_type, 0) // 10 if attacker else 0)

    def move_score(self, board: chess.Board, move: chess.Move, ply: int, ttmove: Optional[chess.Move]) -> int:
        if move == ttmove: return 1_000_000
        if board.is_capture(move):
            victim = board.piece_at(move.to_square) or chess.Piece(chess.PAWN, not board.turn)
            attacker = board.piece_at(move.from_square)
            return 100_000 + 10 * VICTIM[victim.piece_type] - VICTIM.get(attacker.piece_type, 0) + self.see(board, move)
        if move in self.killers[ply]: return 90_000
        return int(self.history[int(board.turn), move.from_square, move.to_square])

    def ordered_moves(self, board: chess.Board, ply: int, ttmove: Optional[chess.Move]) -> List[chess.Move]:
        return sorted(board.legal_moves, key=lambda m: self.move_score(board, m, ply, ttmove), reverse=True)

    def search_root(self, board: chess.Board, seconds: float) -> SearchInfo:
        info = SearchInfo(start=time.time(), limit=seconds)
        book_move = self.book_move(board)
        if book_move:
            info.best = book_move; info.score = self.evaluate_white(board); return info
        alpha, beta, last = -INF, INF, 0
        depth = 1
        while time.time() - info.start < seconds and depth <= 64:
            window = 35
            alpha, beta = last - window, last + window
            while True:
                score, move = self.pvs(board, depth, alpha, beta, 0, info, True)
                if info.stop: break
                if score <= alpha:
                    alpha -= window; window *= 2
                elif score >= beta:
                    beta += window; window *= 2
                else:
                    last = score; info.best = move; info.score = (score if board.turn == chess.WHITE else -score); info.depth = depth; break
            if info.stop: break
            depth += 1
        return info

    def pvs(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, info: SearchInfo, root: bool=False) -> Tuple[int, Optional[chess.Move]]:
        if info.nodes & 2047 == 0 and time.time() - info.start >= info.limit:
            info.stop = True; return 0, None
        info.nodes += 1; info.seldepth = max(info.seldepth, ply)
        if board.is_checkmate(): return -MATE + ply, None
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw(): return 0, None
        if depth <= 0: return self.qsearch(board, alpha, beta, ply, info), None
        in_check = board.is_check()
        key = self.key(board); entry = self.tt.get(key); ttmove = entry.move if entry else None
        if entry and entry.depth >= depth and not root:
            if entry.flag == 0: return entry.score, entry.move
            if entry.flag == 1 and entry.score >= beta: return entry.score, entry.move
            if entry.flag == -1 and entry.score <= alpha: return entry.score, entry.move
        static = self.evaluate_stm(board)
        if not in_check and depth <= 3 and static - 90 * depth >= beta:
            return static, None
        if not in_check and depth >= 3 and abs(static) < MATE // 2:
            board.push(chess.Move.null())
            score, _ = self.pvs(board, depth - 1 - 2, -beta, -beta + 1, ply + 1, info)
            board.pop()
            if info.stop: return 0, None
            if -score >= beta: return beta, None
        best, best_score, old_alpha = None, -INF, alpha
        moves = self.ordered_moves(board, ply, ttmove)
        for i, move in enumerate(moves):
            quiet = not board.is_capture(move) and not board.gives_check(move)
            if quiet and depth <= 2 and not in_check and static + 120 * depth <= alpha:
                continue
            board.push(move)
            reduction = 0
            if quiet and depth >= 3 and i >= 4 and not in_check:
                reduction = 1 + int(depth >= 5 and i >= 8)
            if i == 0:
                score, _ = self.pvs(board, depth - 1, -beta, -alpha, ply + 1, info)
                score = -score
            else:
                score, _ = self.pvs(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, info)
                score = -score
                if score > alpha and reduction:
                    score, _ = self.pvs(board, depth - 1, -alpha - 1, -alpha, ply + 1, info); score = -score
                if alpha < score < beta:
                    score, _ = self.pvs(board, depth - 1, -beta, -alpha, ply + 1, info); score = -score
            board.pop()
            if info.stop: return 0, None
            if score > best_score: best_score, best = score, move
            if score > alpha: alpha = score
            if alpha >= beta:
                if quiet:
                    self.killers[ply][1] = self.killers[ply][0]; self.killers[ply][0] = move
                    self.history[int(board.turn), move.from_square, move.to_square] += depth * depth
                break
        flag = 0 if best_score > old_alpha and best_score < beta else (1 if best_score >= beta else -1)
        self.tt[key] = TTEntry(depth, best_score, flag, best)
        return best_score, best

    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, info: SearchInfo) -> int:
        info.qnodes += 1
        if board.is_check():
            stand = -INF
            caps = list(board.legal_moves)
        else:
            stand = self.evaluate_stm(board)
            if stand >= beta: return beta
            if alpha < stand: alpha = stand
            caps = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        caps.sort(key=lambda m: self.move_score(board, m, ply, None), reverse=True)
        for move in caps:
            if not board.is_check() and self.see(board, move) < -80:
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

    def go(self, board: chess.Board, movetime: float) -> SearchInfo:
        return self.search_root(board, max(0.05, movetime))

def fmt_eval(cp: int) -> str:
    if abs(cp) > MATE - 1000:
        return f"mate {int(math.copysign((MATE - abs(cp) + 1) // 2, cp))}"
    return f"{cp/100:.2f}"

def interactive() -> None:
    engine, board = Engine(), chess.Board()
    color = input("Play as white or black? [w/b]: ").strip().lower().startswith('w')
    tpm = float(input("Time per engine move in seconds: ").strip() or "2")
    print(board)
    while not board.is_game_over():
        if board.turn == color:
            mv = input("Your move (SAN or UCI): ").strip()
            try:
                move = board.parse_san(mv) if not chess.Move.from_uci(mv) in board.legal_moves else chess.Move.from_uci(mv)
            except Exception:
                try: move = chess.Move.from_uci(mv)
                except Exception: print("Invalid move."); continue
            if move not in board.legal_moves: print("Illegal move."); continue
            board.push(move)
        else:
            info = engine.go(board, tpm)
            move = info.best or next(iter(board.legal_moves))
            board.push(move)
            elapsed = max(1e-6, time.time() - info.start)
            nodes = info.nodes + info.qnodes
            print(f"Engine move: {move.uci()}")
            print(f"nodes={nodes} depth={info.depth} seldepth={info.seldepth} nps={int(nodes/elapsed)} time={elapsed:.3f}s eval_white={fmt_eval(info.score)}")
        print(board, "\nFEN:", board.fen())
    print("Game over:", board.result(), board.outcome())

def uci() -> None:
    engine, board = Engine(), chess.Board()
    while True:
        line = sys.stdin.readline()
        if not line: break
        parts = line.strip().split()
        if not parts: continue
        cmd = parts[0]
        if cmd == 'uci':
            print('id name SingleFilePyEngine'); print('id author OpenAI'); print('uciok')
        elif cmd == 'isready': print('readyok')
        elif cmd == 'ucinewgame': board.reset(); engine.tt.clear()
        elif cmd == 'position':
            idx = 1
            if parts[idx] == 'startpos': board = chess.Board(); idx += 1
            elif parts[idx] == 'fen':
                fen = ' '.join(parts[idx+1:idx+7]); board = chess.Board(fen); idx += 7
            if idx < len(parts) and parts[idx] == 'moves':
                for m in parts[idx+1:]: board.push(chess.Move.from_uci(m))
        elif cmd == 'go':
            mt = 1.0
            if 'movetime' in parts: mt = int(parts[parts.index('movetime') + 1]) / 1000
            elif 'wtime' in parts and 'btime' in parts:
                rem = int(parts[parts.index('wtime' if board.turn == chess.WHITE else 'btime') + 1]) / 1000
                mt = max(0.05, rem / 30)
            info = engine.go(board, mt); nodes = info.nodes + info.qnodes
            uci_score = info.score if board.turn == chess.WHITE else -info.score
            print(f"info depth {info.depth} seldepth {info.seldepth} score cp {uci_score} nodes {nodes} nps {int(nodes/max(1e-6,time.time()-info.start))}")
            print('bestmove', (info.best or next(iter(board.legal_moves))).uci())
        elif cmd == 'quit': break
        sys.stdout.flush()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'uci':
        uci()
    else:
        interactive()
