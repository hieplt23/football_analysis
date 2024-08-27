from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssinger
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovement
from view_transformer import ViewTransformer
from speed_and_distance import SpeedAndDistance
import cv2
import numpy as np

def main():
    # read video
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    # initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    
    # get object positions
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovement(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True,
                                                                              stub_path="stubs/camera_movement_stubs.pkl")
    
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)
    
    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # assign player teams
    team_assigner = TeamAssinger()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # interpolate ball position
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # speed and distance estimator
    speed_and_distance = SpeedAndDistance()
    speed_and_distance.add_speed_and_distance_to_tracks(tracks)
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    
    # draw output video
    ## draw speed and distance
    speed_and_distance.draw_speed_and_distance(video_frames, tracks)
    
    ## draw object tracks
    output_video_frames =  tracker.draw_anotations(video_frames, tracks, team_ball_control)
    
    ## draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    
    # save video
    save_video(output_video_frames, "output_videos/output.avi")

if __name__ == '__main__':
    main()