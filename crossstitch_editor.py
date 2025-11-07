import os
import sys
from pathlib import Path
import pygame
from PIL import Image

# Configuration
CELL_SIZE = 20   # pixels per stitch cell
GRID_LINE_COLOR = (40, 40, 40)
BG_COLOR = (255, 255, 255)
SAVE_DIR = Path("edited_outputs")


def load_image_as_grid(image_path):
    """Loads the small pattern image (e.g. 40x40) and returns a 2D array of RGB tuples."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    pixels = list(img.getdata())
    grid = [pixels[i * w:(i + 1) * w] for i in range(h)]
    return grid, w, h


def draw_grid(screen, grid, palette_index):
    """Draws the grid on screen with visible cells and palette highlight."""
    screen.fill(BG_COLOR)
    h = len(grid)
    w = len(grid[0])

    for y in range(h):
        for x in range(w):
            color = grid[y][x]
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

    # Grid lines
    for x in range(w + 1):
        pygame.draw.line(screen, GRID_LINE_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, h * CELL_SIZE))
    for y in range(h + 1):
        pygame.draw.line(screen, GRID_LINE_COLOR, (0, y * CELL_SIZE), (w * CELL_SIZE, y * CELL_SIZE))

    # Display active palette color (in a corner)
    active_color = PALETTE[palette_index]
    pygame.draw.rect(screen, active_color, pygame.Rect(10, 10, 40, 40))
    pygame.display.set_caption(f"Cross Stitch Editor - Active Color {palette_index+1}")


def save_grid_as_image(grid, filename):
    """Save current grid as PNG."""
    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGB", (w, h))
    flat = [c for row in grid for c in row]
    img.putdata(flat)
    SAVE_DIR.mkdir(exist_ok=True)
    path = SAVE_DIR / filename
    img.save(path)
    print(f"[saved] {path}")


def run_editor(image_path):
    grid, w, h = load_image_as_grid(image_path)
    pygame.init()

    screen = pygame.display.set_mode((w * CELL_SIZE, h * CELL_SIZE))
    clock = pygame.time.Clock()

    palette_index = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    filename = Path(image_path).stem + "_edited.png"
                    save_grid_as_image(grid, filename)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    palette_index = event.key - pygame.K_1
                    palette_index = min(palette_index, len(PALETTE) - 1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                gx = x // CELL_SIZE
                gy = y // CELL_SIZE
                if 0 <= gx < w and 0 <= gy < h:
                    if event.button == 1:
                        grid[gy][gx] = PALETTE[palette_index]
                    elif event.button == 3:
                        grid[gy][gx] = (255, 255, 255)  # erase / white

        draw_grid(screen, grid, palette_index)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# Example default palette (expandable)
PALETTE = [
    (0, 0, 0),          # 1 black
    (255, 255, 255),    # 2 white
    (255, 0, 0),        # 3 red
    (0, 255, 0),        # 4 green
    (0, 0, 255),        # 5 blue
    (255, 255, 0),      # 6 yellow
    (255, 128, 0),      # 7 orange
    (255, 0, 255),      # 8 magenta
    (0, 255, 255),      # 9 cyan
]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crossstitch_editor.py path/to/pattern_image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    run_editor(image_path)