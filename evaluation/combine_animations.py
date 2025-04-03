import os
import numpy as np
import cv2
from PIL import Image
import glob

def create_grid_video(input_dir, output_path):
    # Get all GIF files
    gif_files = sorted(glob.glob(os.path.join(input_dir, "anim_A*.gif")))
    
    # Filter out empty GIF files
    gif_files = [f for f in gif_files if os.path.getsize(f) > 0]
    
    if not gif_files:
        print("No valid GIF files found!")
        return
    
    print(f"Found {len(gif_files)} animation files")
    
    # Read first GIF to get dimensions
    first_gif = Image.open(gif_files[0])
    frame_width = 400  # Target width for each frame
    frame_height = 400  # Target height for each frame
    
    # Calculate grid dimensions based on the number of files
    n_rows = 4  # 4 rows for different combinations of A and C
    n_cols = 2  # 2 columns for different theta values
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                         (frame_width * n_cols, frame_height * n_rows))
    
    # Process each frame
    frame_idx = 0
    while True:
        # Create empty grid frame
        grid_frame = np.zeros((frame_height * n_rows, frame_width * n_cols, 3), dtype=np.uint8)
        
        # Read frame from each GIF
        all_frames_read = True
        for i, gif_file in enumerate(gif_files[:n_rows * n_cols]):  # Limit to grid size
            gif = Image.open(gif_file)
            try:
                gif.seek(frame_idx)
                frame = np.array(gif.convert('RGB'))
                # Resize frame
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Calculate position in grid
                row = i // n_cols
                col = i % n_cols
                
                # Place frame in grid
                grid_frame[row*frame_height:(row+1)*frame_height,
                          col*frame_width:(col+1)*frame_width] = frame
            except EOFError:
                all_frames_read = False
                break
        
        if not all_frames_read:
            break
            
        # Write grid frame to video
        out.write(grid_frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
    
    # Release resources
    out.release()
    print("Video creation completed!")
    for gif_file in gif_files:
        Image.open(gif_file).close()

def main():
    # Get the results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    # Output path for the combined video
    output_path = os.path.join(results_dir, "combined_grid.mp4")
    
    # Create the grid video
    create_grid_video(results_dir, output_path)
    print(f"Grid video created and saved to: {output_path}")

if __name__ == "__main__":
    main() 