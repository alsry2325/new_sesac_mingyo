import tkinter as tk
import random
from tkinter import messagebox

# 설정값들
BOARD_SIZE = 15        # 보드 칸 수 (15x15)
CELL_SIZE = 40         # 한 칸의 픽셀 크기
MARGIN = 20            # 보드 바깥 여백
STONE_RADIUS = 14      # 돌 반지름
WINDOW_SIZE = MARGIN*2 + CELL_SIZE*(BOARD_SIZE-1)  # 캔버스 실제 크기


# 전역 상태: 0=빈칸, 1=흑, 2=백
board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
current_player = 1     # 시작은 흑(1)
game_over = False      # 게임 종료 여부

DEBUG_MODE = True   # 점수 시각화 모드 ON/OFF
score_texts = []    # 점수 텍스트 객체 저장용


# Tkinter 창 및 캔버스 생성
root = tk.Tk()
root.title("오목 (Tkinter)")

canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="#F9E6C6")
canvas.grid(row=0, column=0, columnspan=3)

# 상태 레이블(누구 차례인지 표시)
status_var = tk.StringVar()
status_var.set("흑(●) 차례")
status_label = tk.Label(root, textvariable=status_var, font=("Arial", 12))
status_label.grid(row=1, column=0, sticky="w", padx=10, pady=6)

# 새 게임 함수
def reset_game():
    global board, current_player, game_over
    board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    current_player = 1
    game_over = False
    status_var.set("흑(●) 차례")
    canvas.delete("all")
    draw_board()

# 보드(격자) 그리기
def draw_board():
    # 가로/세로 격자선 그리기
    for i in range(BOARD_SIZE):
        x = MARGIN + i * CELL_SIZE
        canvas.create_line(MARGIN, x, WINDOW_SIZE-MARGIN, x, fill="black")
        canvas.create_line(x, MARGIN, x, WINDOW_SIZE-MARGIN, fill="black")
    # 중앙점(optional)
    mid = BOARD_SIZE // 2
    def draw_dot(r, c):
        x = MARGIN + c*CELL_SIZE
        y = MARGIN + r*CELL_SIZE
        canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
    # 3~4개 중앙 포인트만 표시 (전통적)
    points = [3, mid, BOARD_SIZE-4]
    for r in points:
        for c in points:
            draw_dot(r, c)
    # 이미 놓인 돌 다시 그리기
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != 0:
                draw_stone(r, c, board[r][c])

# 특정 칸에 돌 그리기
def draw_stone(row, col, player):
    x = MARGIN + col*CELL_SIZE
    y = MARGIN + row*CELL_SIZE
    color = "black" if player == 1 else "white"
    canvas.create_oval(x-STONE_RADIUS, y-STONE_RADIUS, x+STONE_RADIUS, y+STONE_RADIUS,
                       fill=color, outline="black")

# 좌표(픽셀)를 보드 인덱스로 변환
def pixel_to_index(px, py):
    # 가장 가까운 교차점(격자 교차)을 찾음
    cx = round((px - MARGIN) / CELL_SIZE)
    cy = round((py - MARGIN) / CELL_SIZE)
    if 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
        return cy, cx  # (row, col)
    return None

# 승리 체크: 주어진 마지막 둔 위치에서 5연속인지 확인
def check_win(row, col, player):
    # 4방향 (수평, 수직, 우상향 대각, 우하향 대각)
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    for dr, dc in directions:
        count = 1
        # 한쪽 방향으로 연장
        r, c = row+dr, col+dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r += dr; c += dc
        # 반대쪽 방향으로 연장
        r, c = row-dr, col-dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r -= dr; c -= dc
        if count >= 5:
            return True
    return False

# 클릭 이벤트 처리: 돌 놓기, 턴 전환, 승리 판정
def on_click(event):
    global current_player, game_over
    if game_over:
        return
    idx = pixel_to_index(event.x, event.y)
    if not idx:
        return
    r, c = idx
    if board[r][c] != 0:
        return  # 이미 돌이 있으면 무시
    # 돌 놓기
    board[r][c] = current_player
    draw_stone(r, c, current_player)
    # 승리 체크
    if check_win(r, c, current_player):
        game_over = True
        winner = "흑(●)" if current_player == 1 else "백(○)"
        messagebox.showinfo("승리!", f"{winner}가 이겼습니다!")
        status_var.set(f"{winner} 승리!")
        return
    # 무승부 체크 (판이 다 찬 경우)
    full = all(all(cell != 0 for cell in rowv) for rowv in board)
    if full:
        game_over = True
        messagebox.showinfo("무승부", "판이 가득 찼습니다. 무승부입니다.")
        status_var.set("무승부")
        return
    # 턴 교체
    current_player = 2 if current_player == 1 else 1
    status_var.set("흑(●) 차례" if current_player == 1 else "백(AI) 차례")

    # ✅ AI 턴 자동 실행
    if current_player == 2 and not game_over:
        ai_move()


#ai
def ai_move():
    global current_player, game_over

    best_score = -1
    best_moves = []  # 점수가 같은 칸이 여러 개면 랜덤하게 선택하기 위해

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 0:
                # 공격과 방어 점수 계산
                attack = evaluate_position(r, c, 2)
                defense = evaluate_position(r, c, 1)
                score = attack + defense * 0.8  # 방어는 약간 덜 중요

                # 최고 점수 갱신
                if score > best_score:
                    best_score = score
                    best_moves = [(r, c)]
                elif score == best_score:
                    best_moves.append((r, c))

    # 점수가 가장 높은 위치 중 하나를 랜덤 선택
    if best_moves:
        row, col = random.choice(best_moves)
        board[row][col] = 2  # 백(AI) 돌 두기
        draw_stone(row, col, 2)

        # 승리 판정
        if check_win(row, col, 2):
            game_over = True
            messagebox.showinfo("패배", "AI(백)가 이겼습니다!")
            status_var.set("AI(백) 승리!")
            return

        # 턴 교체
        current_player = 1
        status_var.set("흑(●) 차례")

        highlight_ai_move(row, col)

    # ✅ AI가 둔 이후 점수 시각화 갱신 (자동)
    if DEBUG_MODE and not game_over:
        show_ai_scores(2)


# 점수 평가 함수
def evaluate_position(row, col, player):
    # 임시로 둬보기
    board[row][col] = player

    score = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1  # 자기 자신 포함
        open_ends = 0  # 양쪽이 열려 있는가

        # 한쪽 방향 탐색
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 0:
            open_ends += 1

        # 반대쪽 방향 탐색
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 0:
            open_ends += 1

        # 점수 부여
        if count >= 5:
            score += 10000
        elif count == 4:
            if open_ends == 2:
                score += 5000   # 열린 4
            elif open_ends == 1:
                score += 1000   # 막힌 4
        elif count == 3:
            if open_ends == 2:
                score += 500
            elif open_ends == 1:
                score += 100
        elif count == 2:
            if open_ends == 2:
                score += 50
            elif open_ends == 1:
                score += 10

    # 둔 돌 원상복구
    board[row][col] = 0
    return score
#보드 위 점수 표시 함수 추가
def show_ai_scores(ai_player):
    global score_texts
    for text in score_texts:
        canvas.delete(text)
    score_texts = []

    if not DEBUG_MODE:
        return

    # 1️⃣ 전체 점수 계산
    scores = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 0:
                score = evaluate_position(row, col, ai_player)
                if score > 0:
                    scores.append(score)
    if not scores:
        return

    total_score = sum(scores)
    max_score = max(scores)

    # 2️⃣ 점수를 백분율(%)로 변환해서 표시
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 0:
                score = evaluate_position(row, col, ai_player)
                if score > 0:
                    percentage = (score / total_score) * 100
                    intensity = int((score / max_score) * 255)

                    # 🔥 점수 높을수록 빨강색 진하게 (R,G,B)
                    r = 200 + int(55 * (score / max_score))   # 최대 255
                    g = 50 + int(50 * (1 - score / max_score)) # 낮은 점수일수록 어둡게
                    b = 50
                    color = f"#{r:02x}{g:02x}{b:02x}"

                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    text_id = canvas.create_text(
                        x, y,
                        text=f"{percentage:.1f}%",
                        font=("Arial", 8, "bold"),
                        fill=color
                    )
                    score_texts.append(text_id)
#AI가 둔 자리 강조 표시
def highlight_ai_move(row, col):
    x1, y1 = col * CELL_SIZE, row * CELL_SIZE
    x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
    rect = canvas.create_rectangle(x1+2, y1+2, x2-2, y2-2, outline="red", width=2)
    # 1초 후 자동 삭제
    canvas.after(1000, lambda: canvas.delete(rect))

#단계 (선택) 키보드로 D를 눌러 디버그 모드를 켜고 끄는 기능
def toggle_debug(event):
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    print("DEBUG_MODE =", DEBUG_MODE)
    show_ai_scores(2)


canvas.bind_all("<d>", toggle_debug)
# 새 게임 버튼
reset_button = tk.Button(root, text="새 게임", command=reset_game)
reset_button.grid(row=1, column=1, padx=6)

# 종료 버튼
quit_button = tk.Button(root, text="종료", command=root.quit)
quit_button.grid(row=1, column=2, padx=6)

# 캔버스 클릭 바인딩 및 초기 보드 그리기
canvas.bind("<Button-1>", on_click)
draw_board()

# Tk 이벤트 루프 시작
root.mainloop()