from display import *


import time
def main(env):
    play = True
    game = env  # Initialize the game
    running = True
    clock = pygame.time.Clock()  # Used to control the frame rate
    show_next_piece_info = False
    current_song = random.choice(play_lists)
    pygame.mixer.music.load(current_song)
    pygame.mixer.music.play()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == MUSIC_END_EVENT:
                if play == True:
                    next_song = current_song
                    while next_song == current_song:  # Ensure new song is different
                        next_song = random.choice(play_lists)
                    current_song = next_song
                    pygame.mixer.music.load(current_song)
                    pygame.mixer.music.play()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(game, pygame.mouse.get_pos())  # Handle mouse click
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if current_song is not None:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        play = False
                        current_song = None
                    else:
                        play = True
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        next_song = current_song
                        while next_song == current_song:  # Ensure new song is different
                            next_song = random.choice(play_lists)
                        current_song = next_song
                        pygame.mixer.music.load(current_song)
                        pygame.mixer.music.play()

                if event.key == pygame.K_SPACE:
                    show_next_piece_info = True
                if event.key == pygame.K_h:
                    if game.held_piece == None:
                        game.hold_piece()
                    else:
                        game.unhold_piece()
                if event.key == pygame.K_g:
                    legal_moves = game.generate_legal_moves(game.board, game.current_piece, game.held_piece, game.next_pieces)
                    for x in legal_moves:
                        print(x.piece, x.y, x.x, x.hold)
                        print(x)
                        print(...)
                    print(len(legal_moves))
                if event.key == pygame.K_u:
                    game.undo_move()

          # Show next piece info when space is pressed

        # Draw everything on the screen
            screen.fill(OBSIDIAN)  # Clear the screen with black
            draw_current_piece_label()
            draw_next_piece_label()
            draw_board(env.board)  # Draw the game board
            draw_score(env.score)
            draw_current(env.current_piece)
            draw_next_piece(env.next_pieces[0])
            draw_held_piece_label()
            draw_held_piece(env.held_piece)
            draw_pts_per_move(env.pts_per_move)
        if show_next_piece_info:
            game.display_board() 
            display_next_piece_info(game)
            display_current_piece_info(game)
            print(f"The points:{game.score}")
            print(f"pts per move {game.pts_per_move}")
            print("."*10)
              # Display next piece info
            show_next_piece_info = False  # Reset flag after displaying info
        pygame.display.flip()  # Update the screen
        clock.tick(30)  # Limit to 30 frames per second

    pygame.quit()  # Quit the game when the loop end
test = Game()

main(test)


