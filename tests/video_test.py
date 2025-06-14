import gradio as gr
import numpy as np

def process(video_path, audio_data):
    """
    Echo back the recorded video file path and the recorded audio array.
    - video_path: local path to the recorded video file (mp4/webm)
    - audio_data: tuple (sample_rate, np.ndarray) from microphone recording
    """
    return video_path, audio_data

def main():
    with gr.Blocks(title="Gradio Recording Demo") as demo:
        gr.Markdown("**Record video and audio, then click Submit to play back.**")
        with gr.Row():
            video_input = gr.Video(sources="webcam", label="Record a short video")
            audio_input = gr.Audio(sources="microphone", type="numpy", label="Record audio")
        with gr.Row():
            video_output = gr.Video(label="Playback Video")
            audio_output = gr.Audio(label="Playback Audio")
        submit = gr.Button("Submit")
        submit.click(
            fn=process,
            inputs=[video_input, audio_input],
            outputs=[video_output, audio_output]
        )
    demo.launch()

if __name__ == "__main__":
    main()