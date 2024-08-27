from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from utils import get_bbox_width, get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                        
                    tracks[object][frame_num][track_id]['position'] = position
                        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", {}) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing value
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:{"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1 )
            detections += detections_batch
            
        return detections
    
    def draw_circular_id(self, frame, x_center, y2, track_id, color):
        radius = 20
        cv2.circle(frame, (x_center, y2 + radius), radius, color, -1)
        cv2.circle(frame, (x_center, y2 + radius), radius, (0, 0, 0), 2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(str(track_id), font, 0.7, 2)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y2 + radius + text_size[1] // 2
        
        cv2.putText(frame, str(track_id), (text_x, text_y), font, 0.7, (0, 0, 0), 2)
        
    def draw_ellipse(self, frame, bbox, color, track_id=None, has_ball=False):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        if has_ball:
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width)+5, int(0.35*(width+5))),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=(0, 0, 255),
                lineType=cv2.LINE_8, 
                thickness=2
            )

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            lineType=cv2.LINE_8, 
            thickness=2
        )

        retangle_width = 40
        retangle_height = 20
        x1_rect = x_center - retangle_width//2
        x2_rect = x_center + retangle_width//2
        y1_rect = (y2 - retangle_height//2) + 15
        y2_rect = (y2 + retangle_height//2) + 15

        if track_id is not None:
            self.draw_circular_id(frame, x_center, y2, track_id, color)
            
            # cv2.rectangle(frame,
            #               (int(x1_rect), int(y1_rect)),
            #               (int(x2_rect), int(y2_rect)),
            #               color=color,
            #               thickness=-1
            # )
            # x1_text = x1_rect + 12
            # if track_id > 99:
            #     x1_text -= 10
            # cv2.putText(
            #     frame,
            #     f"{track_id}",
            #     (int(x1_text), int(y1_rect + 17)),
            #     cv2.FONT_HERSHEY_DUPLEX,
            #     color=(0, 0, 0),
            #     fontScale=0.7,
            #     thickness=2
            # )

        return frame
    
    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        
        traingle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [traingle_points], 0, color, thickness=-1)
        cv2.drawContours(frame, [traingle_points], 0, (0, 0, 0), thickness=2)
        
        return frame
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
                
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_name_inv = {v:k for k, v in cls_names.items()}
            
            # convert  to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # convert goalkeeper to player object
            for id, class_name in enumerate(detection_supervision.class_id):
                if cls_names[class_name] == "goalkeeper":
                    detection_supervision.class_id[id] = cls_name_inv["player"]
                    
            # track object
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_name_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == cls_name_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3] 
                    
                    if cls_id == cls_name_inv["ball"]:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)        
                    
        return tracks
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 870), (440, 1020), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        
        total_frames = team_1_num_frames + team_2_num_frames
        team_1_ratio = team_1_num_frames / total_frames
        team_2_ratio = team_2_num_frames / total_frames

        # Draw bar chart
        bar_height = 50
        bar_width_team_1 = int(400 * team_1_ratio)
        bar_width_team_2 = int(400 * team_2_ratio)

        # Draw Team 1 bar
        cv2.rectangle(frame, (25, 890), (10 + bar_width_team_2, 890 + bar_height), (153, 153, 255), -1)
        
        # Draw Team 2 bar
        cv2.rectangle(frame, (25, 950), (10 + bar_width_team_1, 950 + bar_height), (153, 255, 153), -1)

        # Add text labels
        cv2.putText(frame, f"Team 1: {team_2_ratio * 100:.2f}%", (30, 925), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {team_1_ratio * 100:.2f}%", (30, 985), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

        return frame
    
    def draw_anotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # draw player
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id, player.get("has_ball", False))
                
            # draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
                
            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 0, 255))
                
            # draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
                
            output_video_frames.append(frame)
            
        return output_video_frames