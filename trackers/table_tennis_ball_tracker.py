from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class TableTennisBallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # interpolate the missing values (more aggressive for faster table tennis ball)
        df_ball_positions = df_ball_positions.interpolate(method='linear')
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_x'] = (df_ball_positions['x1'] + df_ball_positions['x2'])/2
        
        # Use smaller rolling window for faster table tennis action
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=3, min_periods=1, center=False).mean()
        df_ball_positions['mid_x_rolling_mean'] = df_ball_positions['mid_x'].rolling(window=3, min_periods=1, center=False).mean()
        
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['delta_x'] = df_ball_positions['mid_x_rolling_mean'].diff()
        
        # Reduce minimum change frames for faster table tennis
        minimum_change_frames_for_hit = 8  # Reduced from 25 for faster game
        
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit*1.2)):
            # Check for direction changes in both x and y
            negative_y_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_y_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0
            
            # Also consider rapid x direction changes (side-to-side shots)
            negative_x_change = abs(df_ball_positions['delta_x'].iloc[i]) > 2  # Threshold for significant x movement
            
            if negative_y_change or positive_y_change or negative_x_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    if change_frame >= len(df_ball_positions):
                        break
                        
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_y_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_y_change and positive_position_change_following_frame:
                        change_count += 1
            
                if change_count > minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        # Lower confidence threshold for smaller table tennis ball
        results = self.model.predict(frame, conf=0.10)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"TT Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10 )), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                # Smaller rectangle for table tennis ball
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


# Alias for backward compatibility  
BallTracker = TableTennisBallTracker