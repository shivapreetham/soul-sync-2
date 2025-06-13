import gradio as gr

def process(video_path, audio_data):
    """
    Echo back the recorded video file path and the recorded audio array.
    - video_path: local path to the recorded video file (mp4/webm)
    - audio_data: tuple (sample_rate, np.ndarray) from microphone recording
    """
    # For demo, we just return them so Gradio shows playback widgets.
    return video_path, audio_data

def main():
    # Using legacy inputs/outputs so it works on Gradio v5.33.2
    video_input = gr.inputs.Video(source="webcam", label="Record a short video")
    audio_input = gr.inputs.Audio(source="microphone", type="numpy", label="Record audio")
    video_output = gr.outputs.Video(label="Playback Video")
    audio_output = gr.outputs.Audio(label="Playback Audio")

    iface = gr.Interface(
        fn=process,
        inputs=[video_input, audio_input],
        outputs=[video_output, audio_output],
        title="Gradio Recording Demo (legacy API)",
        description="Record video and audio, then playback."
    )
    iface.launch()

if __name__ == "__main__":
    main()
