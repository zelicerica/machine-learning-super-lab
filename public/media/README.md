# Neural-network motion background

`neural-network.mp4` is the 8-second HeyGen clip selected for the Super Lab's moving background. It depicts translucent connected blocks with cyan, violet, and mint light. It is abstract decorative imagery, not a visualization of actual model weights, topology, or activations.

The accepted source was generated with HeyGen Video 1 (video ID `87135aee862845578cb6feb463fffb0c`). The bundled derivative preserves its sequence and duration, scales it to 1280 × 732 at 24 frames per second, removes audio, and places the MP4 metadata at the beginning for browser playback. H.264 video with 4:2:0 pixels is used. The video is approximately 1.2 MB; no external player or network connection is needed after downloading the lab.

`neural-network-poster.jpg` is a frame from the same clip at three seconds. It provides a still background while the video loads or when playback is unavailable.

Playback is muted and looped. The visible Pause motion / Play motion button stores the preference in this browser. Reduced-motion preferences suppress automatic playback unless the learner explicitly opts in. Hidden browser tabs pause playback. One video element persists across lesson navigation.

To reproduce the web encoding from the original generated source using FFmpeg:

```text
ffmpeg -i source.mp4 -vf scale=1280:-2 -c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -an -movflags +faststart neural-network.mp4
ffmpeg -ss 3 -i neural-network.mp4 -frames:v 1 -q:v 3 neural-network-poster.jpg
```
