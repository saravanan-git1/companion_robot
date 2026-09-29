import numpy as np
import cv2

class RobotFace:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        
    def render_face(self, mood, frame_count):
        """Generates a visual matrix representation of the robot's facial state."""
        # Create a blank black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if mood == "Idle 😐":
            # Normal cyan eyes that blink periodically
            color = (255, 255, 0) # Cyan in BGR
            if frame_count % 25 == 0 or frame_count % 25 == 1:
                # Blinking state (flat lines)
                cv2.line(canvas, (80, 200), (160, 200), color, 8)
                cv2.line(canvas, (240, 200), (320, 200), color, 8)
            else:
                # Open eyes
                cv2.circle(canvas, (120, 200), 35, color, -1)
                cv2.circle(canvas, (280, 200), 35, color, -1)
                
        elif mood == "Happy 😊":
            # Green smiling arc shapes
            color = (0, 255, 127) # Spring Green in BGR
            # Draw inverted semi-circles for happy eyes
            cv2.ellipse(canvas, (120, 210), (35, 30), 0, 180, 360, color, 8)
            cv2.ellipse(canvas, (280, 210), (35, 30), 0, 180, 360, color, 8)
            
        elif mood == "Thinking 🤔":
            # Pulsing orange rings indicating calculations
            color = (0, 140, 255) # Orange in BGR
            pulse_radius = 25 + int((frame_count % 5) * 4)
            cv2.circle(canvas, (120, 200), pulse_radius, color, 6)
            cv2.circle(canvas, (280, 200), pulse_radius, color, 6)
            
        elif mood == "Talking 💬":
            # Yellow wide expressive eyes
            color = (0, 255, 255) # Yellow in BGR
            talk_bounce = int((frame_count % 2) * 5)
            cv2.circle(canvas, (120, 200), 35 + talk_bounce, color, -1)
            cv2.circle(canvas, (280, 200), 35 + talk_bounce, color, -1)
            
        # Add text overlay showing active state on the screen
        cv2.putText(canvas, f"State: {mood}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return canvas
