from sklearn.cluster import KMeans

class TeamAssinger:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # reshape the image to 2D array
        image_2d = image.reshape(-1, 3)
        
        # preform k-means with 2 cluster
        k_means = KMeans(n_clusters=2, random_state=0).fit(image_2d)
        
        return k_means
    
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_img = image[:image.shape[0]//2, :]
        
        # get clustering model
        kmeans = self.get_clustering_model(top_half_img)
        
        # get the cluster labels for each pixel
        labels = kmeans.labels_
        
        # reshape the label to the image shape
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])
        
        # get the player cluster
        corner_clusters = [
            clustered_img[0, 0],
            clustered_img[0, -1],
            clustered_img[-1, 0],
            clustered_img[-1, -1]
        ]
        
        player_cluster = max(corner_clusters, key=corner_clusters.count)
        player_cluster = 1 - player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(player_colors)
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
        self.kmeans = kmeans
        return player_colors
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        
        if player_id == 104:
            team_id = 2
        
        self.player_team_dict[player_id] = team_id
        
        return team_id
        