# System Limitations & Safety Disclaimers

## Known Technical Limitations

1. **2D Camera Geometry:** Pose extraction relies on 2D perspective projection. Extreme camera tilt or steep overhead angles alter body proportions and affect angle calculation accuracy.
2. **Keypoint Occlusion:** Severe occlusion (e.g. person behind solid obstacles or heavy blankets) reduces pose keypoint confidence below threshold.
3. **Low-Light Jitter:** Low lux environments introduce spatial jitter in keypoint coordinates.
4. **Multi-Person Interlocking:** When two people closely cross paths, bounding box IoU overlap may cause short tracking ID switches.

## Important Safety Disclaimers

> [!WARNING]
> This system is designed as an assistive monitoring prototype. It is **NOT** a certified medical device, critical life-safety system, or primary fall prevention system. Do **NOT** rely on this software as a sole monitoring mechanism for high-risk individuals without professional human oversight.
