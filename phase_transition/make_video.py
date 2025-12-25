"""
Create video from animation frames.

Uses ffmpeg or imageio as fallback.
"""

import subprocess
from pathlib import Path
import argparse


def make_video_ffmpeg(frames_dir, output_path, framerate=30):
    """Create video using ffmpeg (best quality)."""
    frames_pattern = Path(frames_dir) / "frame_%05d.png"
    
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(framerate),
        '-i', str(frames_pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        str(output_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Video saved to: {output_path}")
    return True


def make_video_imageio(frames_dir, output_path, framerate=30):
    """Create video using imageio (fallback, lower quality)."""
    try:
        import imageio
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Please install imageio and pillow: pip install imageio pillow imageio-ffmpeg")
        return False
    
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.png"))
    
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    print(f"Creating video with imageio...")
    
    # Read first frame to get dimensions
    first_frame = np.array(Image.open(frame_files[0]))
    
    writer = imageio.get_writer(str(output_path), fps=framerate, quality=8)
    
    for frame_file in frame_files:
        frame = np.array(Image.open(frame_file))
        writer.append_data(frame)
    
    writer.close()
    print(f"Video saved to: {output_path}")
    return True


def make_gif(frames_dir, output_path, framerate=15, optimize=True):
    """Create GIF from frames."""
    try:
        from PIL import Image
    except ImportError:
        print("Please install pillow: pip install pillow")
        return False
    
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.png"))
    
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    print(f"Creating GIF...")
    
    # Load frames
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        if optimize:
            # Reduce colors for smaller file size
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        frames.append(img)
    
    # Save as GIF
    duration = int(1000 / framerate)  # ms per frame
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=optimize
    )
    
    print(f"GIF saved to: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from animation frames")
    parser.add_argument("frames_dir", type=str, help="Directory containing frames")
    parser.add_argument("--output", type=str, default="grokking_animation.mp4",
                        help="Output video path")
    parser.add_argument("--framerate", type=int, default=30, help="Frames per second")
    parser.add_argument("--gif", action="store_true", help="Create GIF instead of video")
    parser.add_argument("--use_imageio", action="store_true", 
                        help="Use imageio instead of ffmpeg")
    
    args = parser.parse_args()
    
    if args.gif:
        output_path = args.output.replace('.mp4', '.gif')
        make_gif(args.frames_dir, output_path, args.framerate)
    elif args.use_imageio:
        make_video_imageio(args.frames_dir, args.output, args.framerate)
    else:
        success = make_video_ffmpeg(args.frames_dir, args.output, args.framerate)
        if not success:
            print("\nFalling back to imageio...")
            make_video_imageio(args.frames_dir, args.output, args.framerate)
