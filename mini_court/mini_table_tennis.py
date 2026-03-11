import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniTableTennis():
    def __init__(self, frame):
        self.drawing_rectangle_width = 200
        self.drawing_rectangle_height = 300
        self.buffer = 50
        self.padding_table = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_table_position()
        self.set_table_drawing_key_points()
        self.set_table_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.TABLE_WIDTH,
                                                self.table_drawing_width
                                            )

    def set_table_drawing_key_points(self):
        # 8 keypoints for table tennis: 4 corners + 2 center line + 2 net
        drawing_key_points = [0]*16

        # Table corners
        # point 0 (top-left)
        drawing_key_points[0], drawing_key_points[1] = int(self.table_start_x), int(self.table_start_y)
        # point 1 (top-right) 
        drawing_key_points[2], drawing_key_points[3] = int(self.table_end_x), int(self.table_start_y)
        # point 2 (bottom-left)
        drawing_key_points[4] = int(self.table_start_x)
        drawing_key_points[5] = self.table_start_y + self.convert_meters_to_pixels(constants.TABLE_DEPTH)
        # point 3 (bottom-right)
        drawing_key_points[6] = int(self.table_end_x)
        drawing_key_points[7] = self.table_start_y + self.convert_meters_to_pixels(constants.TABLE_DEPTH)

        # Center line points
        # point 4 (center line top)
        drawing_key_points[8] = int(self.table_start_x + self.table_drawing_width/2)
        drawing_key_points[9] = int(self.table_start_y)
        # point 5 (center line bottom)  
        drawing_key_points[10] = int(self.table_start_x + self.table_drawing_width/2)
        drawing_key_points[11] = int(self.table_start_y + self.convert_meters_to_pixels(constants.TABLE_DEPTH))

        # Net points (halfway between top and bottom)
        net_y = int(self.table_start_y + self.convert_meters_to_pixels(constants.TABLE_DEPTH/2))
        # point 6 (net left)
        drawing_key_points[12] = int(self.table_start_x)
        drawing_key_points[13] = net_y
        # point 7 (net right)
        drawing_key_points[14] = int(self.table_end_x)
        drawing_key_points[15] = net_y

        self.drawing_key_points = drawing_key_points

    def set_table_lines(self):
        # Define lines connecting keypoints
        self.lines = [
            # Table outline
            (0, 1),  # top edge
            (1, 3),  # right edge  
            (3, 2),  # bottom edge
            (2, 0),  # left edge
            # Center line
            (4, 5),  # center line vertical
            # Net line
            (6, 7),  # net horizontal
        ]

    def set_mini_table_position(self):
        self.table_start_x = self.start_x + self.padding_table
        self.table_start_y = self.start_y + self.padding_table
        self.table_end_x = self.end_x - self.padding_table
        self.table_end_y = self.end_y - self.padding_table
        self.table_drawing_width = self.table_end_x - self.table_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_table(self, frame):
        # Draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Draw table lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            if line == (6, 7):  # Net line
                cv2.line(frame, start_point, end_point, (255, 0, 0), 3)  # Blue and thicker for net
            else:
                cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_table(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.table_start_x, self.table_start_y)
    
    def get_width_of_mini_court(self):
        return self.table_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index, 
                                   player_height_in_pixels, player_height_in_meters):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                          player_height_in_meters,
                                                                          player_height_in_pixels)
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = (self.drawing_key_points[closest_key_point_index*2],
                                      self.drawing_key_points[closest_key_point_index*2+1])
        
        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
                                     closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)

        return mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_table_key_points):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), 
                                          key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoint in pixels (use table corners: 0,1,2,3)
                closest_key_point_index = get_closest_keypoint_index(foot_position, original_table_key_points, [0,1,2,3])
                closest_key_point = (original_table_key_points[closest_key_point_index*2], 
                                     original_table_key_points[closest_key_point_index*2+1])

                # Get player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id])
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get the closest keypoint in pixels for ball
                    closest_key_point_index = get_closest_keypoint_index(ball_position, original_table_key_points, [0,1,2,3])
                    closest_key_point = (original_table_key_points[closest_key_point_index*2], 
                                        original_table_key_points[closest_key_point_index*2+1])
                    
                    mini_court_ball_position = self.get_mini_court_coordinates(ball_position,
                                                                              closest_key_point, 
                                                                              closest_key_point_index, 
                                                                              max_player_height_in_pixels,
                                                                              player_heights[player_id])
                    output_ball_boxes.append({1: mini_court_ball_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes
    
    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 3, color, -1)
        return frames


# Alias for backward compatibility  
MiniCourt = MiniTableTennis