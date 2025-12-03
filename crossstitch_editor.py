import pygame
import sys
import os
from pathlib import Path
from PIL import Image

# Configuration of the grid representing the patter 
MIN_GRID = 20
MAX_GRID = 100

PALETTE = [ # The colours the user can draw with on the UI
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (128, 64, 0),
    (180, 180, 180)
]

SAVE_DIR = Path("edited_outputs")


def load_image(image_path, grid_size): # loading the image already 'preprocessed' as a crossstich pattern
    img = Image.open(image_path).convert("RGB")
    resized = img.resize((grid_size, grid_size), Image.NEAREST)
    pix = list(resized.getdata())
    grid = [pix[i * grid_size:(i + 1) * grid_size] for i in range(grid_size)]
    return grid


def save_grid(grid, filename): # transform the image into a grid representation
    size = len(grid)
    img = Image.new("RGB", (size, size))
    flat = [c for row in grid for c in row]
    img.putdata(flat)

    SAVE_DIR.mkdir(exist_ok=True)
    path = SAVE_DIR / filename
    img.save(path)
    print(f"[Saved] {path}")



def run_editor(image_path): # the main function for starting the UI and recording user commands to edit
    pygame.init()
    PANEL_W = 220
    GRID_PIXELS = 700   

    window = pygame.display.set_mode((GRID_PIXELS + PANEL_W, GRID_PIXELS + 40))
    pygame.display.set_caption("Cross-Stitch Editor")
    font = pygame.font.SysFont("arial", 20)
    smallfont = pygame.font.SysFont("arial", 16)


    grid_size = 50 # initial grid resolution (medium value since 20 is min and 100 max)
    grid = load_image(image_path, grid_size)
    undo_stack = [] # recording the user edits for the Undo button

    active_color = PALETTE[0]

    dragging_slider = False

    running = True
    while running:
        window.fill((240, 240, 240))

        # checking input events:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # check if the user clicked on a colour button
                for i, col in enumerate(PALETTE):
                    bx = GRID_PIXELS + 20
                    by = 20 + i * 38
                    if pygame.Rect(bx, by, 40, 30).collidepoint((mx, my)):
                        active_color = col # draw with that colour

                # undo button of the last move
                if pygame.Rect(GRID_PIXELS + 20, 520, 150, 40).collidepoint((mx, my)):
                    if undo_stack:
                        grid = undo_stack.pop()

                # save button to store the edited image
                if pygame.Rect(GRID_PIXELS + 20, 570, 150, 40).collidepoint((mx, my)):
                    save_grid(grid, Path(image_path).stem + "_edited.png")

                #check if the slider that changes resolution was moved left or right
                sx, sy = GRID_PIXELS + 20, 480
                if sx <= mx <= sx + 160 and sy - 5 <= my <= sy + 5:
                    dragging_slider = True

                # Draw on grid
                if mx < GRID_PIXELS and my < GRID_PIXELS:
                    cell = GRID_PIXELS // grid_size
                    gx = mx // cell
                    gy = my // cell

                    if 0 <= gx < grid_size and 0 <= gy < grid_size:
                        undo_stack.append([row[:] for row in grid])
                        grid[gy][gx] = active_color

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False

            elif event.type == pygame.MOUSEMOTION and dragging_slider:
                mx, my = event.pos
                ratio = min(max((mx - (GRID_PIXELS + 20)) / 160, 0), 1)
                new_size = int(MIN_GRID + ratio * (MAX_GRID - MIN_GRID))

                if new_size != grid_size:
                    grid_size = new_size
                    grid = load_image(image_path, grid_size)

   
        # actual drawing 
        cell = GRID_PIXELS // grid_size
        for y in range(grid_size):
            for x in range(grid_size):
                pygame.draw.rect(window, grid[y][x],
                                 (x * cell, y * cell, cell, cell))

        # grid lines
        for i in range(grid_size + 1):
            pygame.draw.line(window, (50, 50, 50), (i * cell, 0), (i * cell, GRID_PIXELS))
            pygame.draw.line(window, (50, 50, 50), (0, i * cell), (GRID_PIXELS, i * cell))

# UI side pannel (color button, undo button, resolution slider, save button)

        for i, col in enumerate(PALETTE): 
            bx = GRID_PIXELS + 20
            by = 20 + i * 38
            pygame.draw.rect(window, col, (bx, by, 40, 30))  # add colour buttons

            if col == active_color:
                pygame.draw.rect(window, (255, 0, 0), (bx - 2, by - 2, 44, 34), 2)

        window.blit(font.render("Resolution:", True, (0, 0, 0)), (GRID_PIXELS + 20, 450)) # add slider bar:
        pygame.draw.line(window, (100, 100, 100),
                         (GRID_PIXELS + 20, 480),
                         (GRID_PIXELS + 180, 480), 3)

        # slider knob
        ratio = (grid_size - MIN_GRID) / (MAX_GRID - MIN_GRID)
        knob_x = GRID_PIXELS + 20 + int(ratio * 160)
        pygame.draw.circle(window, (0, 0, 0), (knob_x, 480), 8)
        window.blit(smallfont.render(f"{grid_size} x {grid_size}", True, (0, 0, 0)),
                    (GRID_PIXELS + 20, 495))

        # Undo button
        pygame.draw.rect(window, (220, 220, 220), (GRID_PIXELS + 20, 520, 150, 40))
        window.blit(font.render("Undo", True, (0, 0, 0)), (GRID_PIXELS + 70, 530))

        # Save button
        pygame.draw.rect(window, (220, 220, 220), (GRID_PIXELS + 20, 570, 150, 40))
        window.blit(font.render("Save", True, (0, 0, 0)), (GRID_PIXELS + 70, 580))

        pygame.display.flip()

    pygame.quit()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python editor.py input_image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("Image not found.")
        sys.exit(1)

    run_editor(img_path)
