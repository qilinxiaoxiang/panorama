import time
from multiprocessing import Process
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTModel, ViTFeatureExtractor

from sklearn.metrics import adjusted_rand_score
import networkx as nx
from anytree import Node, RenderTree, LevelOrderIter, findall
from PIL import Image
import cv2
import os
import numpy as np


class ImageClusterer:
    def __init__(self, groups, model_name='google/vit-base-patch16-224', similarity_threshold=0.35, sift_threshold=0.6, correspondences_threshold=20):
        self.image_path_indeces = {}
        self.image_paths = []
        self.model = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold
        self.sift_threshold = sift_threshold
        self.correspondences_threshold = correspondences_threshold
        self.graph = nx.Graph()
        self.trees = []
        self.matches_dict = {}
        self.homographies = {}

        # self.groups = [20, 12, 8]
        self.ground_truth = []
        index = 0
        for length in self.groups:
            for i in range(length):
                self.ground_truth.append(index)
            index += 1

    def extract_features(self, images):
        # get the matrix attribute of each image in images
        inputs = self.feature_extractor(images=[x.matrix for x in images], return_tensors="pt")
        outputs = self.model(**inputs)

        # Extract [CLS] token embeddings as image features
        features = outputs.last_hidden_state[:, 0, :]
        return features.detach().numpy()

    def calculate_similarity(self, features):
        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(features)
        return sim_matrix

    def create_similarity_graph(self, sim_matrix, images):
        # Create a graph based on similarity above a threshold
        count = 0
        print(f"len of sim_matrix: {len(sim_matrix)}")
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                # if sim_matrix[i, j] > self.similarity_threshold and self.match_features(images[i], images[j], sim_matrix[i, j]):
                if sim_matrix[i, j] > self.similarity_threshold:
                    count += 1
                    if self.match_features(images[i], images[j], sim_matrix[i, j]):
                        self.graph.add_edge(images[i], images[j], weight=sim_matrix[i, j])
        print(f"compare counts: {count}")


    def match_features(self, image1, image2, similarity):
        # Initialize FLANN based matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Increase for higher accuracy
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Perform matching from image1 to image2
        matches1to2 = self.find_good_matches(flann, image1.descriptors, image2.descriptors)
        # Perform matching from image2 to image1
        matches2to1 = self.find_good_matches(flann, image2.descriptors, image1.descriptors)

        # Retain only those matches that are mutual
        good_matches = [m1 for m1 in matches1to2 if
                        any(m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx for m2 in matches2to1)]

        if len(good_matches) >= self.correspondences_threshold:
            # print('Good matches:', image1.name, image2.name, len(good_matches), similarity)
            self.matches_dict[(image1.name, image2.name)] = good_matches
            self.matches_dict[(image2.name, image1.name)] = [m2 for m2 in matches2to1 if
                        any(m2.queryIdx == m1.trainIdx and m2.trainIdx == m1.queryIdx for m1 in matches1to2)]
            # draw
            # img1 = cv2.imread(image1.path)
            # img2 = cv2.imread(image2.path)
            # # Create a new output image that concatenates the two images together
            # (h1, w1) = img1.shape[:2]
            # (h2, w2) = img2.shape[:2]
            # vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
            # vis[0:h1, 0:w1] = img1
            # vis[0:h2, w1:w1 + w2] = img2
            #
            # # Draw the matches
            # for m in good_matches:
            #     # Draw the match
            #     pt1 = (int(image1.key_points[m.queryIdx].pt[0]), int(image1.key_points[m.queryIdx].pt[1]))
            #     pt2 = (int(image2.key_points[m.trainIdx].pt[0] + w1), int(image2.key_points[m.trainIdx].pt[1]))
            #     cv2.line(vis, pt1, pt2, (255, 0, 0), 1)
            #     cv2.circle(vis, pt1, 5, (0, 0, 255), -1)
            #     cv2.circle(vis, pt2, 5, (0, 255, 0), -1)
            #
            # # Display the combined image with matches
            # cv2.imshow(f'Matches between image {image1.name} and image {image2.name}', vis)
            # cv2.waitKey(0)  # Wait for a key press to move to the next pair
            # cv2.destroyAllWindows()

            return True
        else:
            return False

    def find_good_matches(self, flann, descriptors1, descriptors2):
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        # Filter matches using the Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.sift_threshold * n.distance:
                good_matches.append(m)
        return good_matches

    def calculate_depths(self, root):
        max_depth = 0
        node_depth = {}
        for node in LevelOrderIter(root):
            if node.is_root:
                node_depth[node] = 0
            else:
                node_depth[node] = node_depth[node.parent] + 1
            max_depth = max(max_depth, node_depth[node])
        return node_depth, max_depth

    def find_new_root(self, root):
        node_depth, max_depth = self.calculate_depths(root)
        best_root = root
        best_max_depth = max_depth

        for node in LevelOrderIter(root):
            current_depth = node_depth[node]
            subtree_max_depth = current_depth

            # Check potential new depth if this node were the root
            for child in LevelOrderIter(node):
                potential_depth = node_depth[child] - current_depth + node_depth.get(child.parent, 0)
                subtree_max_depth = max(subtree_max_depth, potential_depth)

            if subtree_max_depth < best_max_depth:
                best_root = node
                best_max_depth = subtree_max_depth

        return best_root, node_depth

    def cluster_images(self, image_paths):
        features = self.extract_features(image_paths)
        sim_matrix = self.calculate_similarity(features)
        self.create_similarity_graph(sim_matrix, image_paths)
        return self.graph

    def find_most_central_node(self, component):
        # Calculate the closeness centrality for each node within the component
        subgraph = self.graph.subgraph(component)
        closeness = nx.closeness_centrality(subgraph, distance='weight')

        # Find the node with the highest closeness centrality in this component
        most_central_node = max(closeness, key=closeness.get)
        return most_central_node

    def construct_trees_from_components(self):
        # Find all connected components of the graph
        components = nx.connected_components(self.graph)

        # Process each component separately
        for component in components:
            central_node = self.find_most_central_node(component)
            root = Node(central_node.name, image=central_node)
            node_mapping = {central_node: root}

            # Use BFS to traverse and build the tree from the most central node
            for start, end in nx.bfs_edges(self.graph.subgraph(component), central_node):
                parent_node = node_mapping[start]
                child_node = Node(end.name, parent=parent_node, image=end)
                node_mapping[end] = child_node

            self.trees.append(root)

        return self.trees

    def display_trees(self, trees):
        for tree_root in trees:
            print("\nTree Rooted at:", tree_root.name)
            for pre, _, node in RenderTree(tree_root):
                print(f"{pre}{node.name}")

    def calculate_homography_to_root(self):
        # Recursive function to calculate homography
        def recurse(node, parent_homography):
            for child in node.children:
                matches = self.matches_dict.get((node.name, child.name))
                src_pts = np.float32([node.image.key_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([child.image.key_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                cumulative_homography = np.dot(parent_homography, np.linalg.inv(H))

                self.homographies[child.name] = cumulative_homography
                recurse(child, cumulative_homography)

        # Start the recursion for each tree root
        for tree_root in self.trees:
            self.homographies[tree_root.name] = np.eye(3)
            recurse(tree_root, self.homographies[tree_root.name])

    def stitch_images_with_perspective(self, root, max_depth):
        images = []
        homographies = []

        # Collect images and their homographies
        for node in findall(root, filter_=lambda n: n.depth <= max_depth):
            if node.image.name in self.homographies:
                img = cv2.imread(node.image.path)  # Load the image
                H = self.homographies[node.name]
                images.append(img)
                homographies.append(H)

        # Determine the size of the output canvas
        min_x, min_y = 0, 0
        max_x, max_y = 0, 0
        for img, H in zip(images, homographies):
            (h, w) = img.shape[:2]
            corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32).reshape(-1, 1, 2)
            # Warp the corners of the current image
            warped_corners = cv2.perspectiveTransform(corners, H)

            x_coords, y_coords = warped_corners[:, 0, 0], warped_corners[:, 0, 1]
            min_x, min_y = min(min_x, min(x_coords)), min(min_y, min(y_coords))
            max_x, max_y = max(max_x, max(x_coords)), max(max_y, max(y_coords))
        # Create the canvas with the computed size
        canvas_width = int(np.ceil(max_x - min_x))
        canvas_height = int(np.ceil(max_y - min_y))
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        # Warp each image onto the canvas
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])  # Translation matrix
        for img, H in zip(images, homographies):
            transformed_homography = translation @ H  # Apply translation
            warped_img = cv2.warpPerspective(img, transformed_homography, (canvas_width, canvas_height))
            warped_mask = cv2.warpPerspective(np.ones(img.shape[:2], dtype=np.uint8), transformed_homography,
                                              (canvas_width, canvas_height))
            non_overlap = warped_mask > mask
            canvas[non_overlap] = warped_img[non_overlap]
            mask = np.maximum(mask, warped_mask)
        cv2.imwrite('../result/' + root.name + '.jpg', canvas)
        return canvas

    def load_images_from_folder(self, folder_path):
        # Supported image formats
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # List all files in the directory
        files = os.listdir(folder_path)

        # Filter and load images
        self.image_paths = [os.path.join(folder_path, file) for file in files if
                       os.path.splitext(file)[1].lower() in valid_extensions]
        self.image_paths.sort()
        index = 0
        for image_path in self.image_paths:
            self.image_path_indeces[image_path] = index
            index += 1
        return [MyImage(x) for x in self.image_paths]

    def measure_grouping(self):
        # define predictions as a 140 long list
        predictions = [-1] * sum(self.groups)
        index = 0
        for tree in self.trees:
            # traverse each tree, assign the same group number to each image in the tree
            for node in LevelOrderIter(tree):
                predictions[self.image_path_indeces[node.image.path]] = index
            index += 1
        print(self.ground_truth)
        print(predictions)
        print(self.similarity_threshold, self.sift_threshold, self.correspondences_threshold, adjusted_rand_score(self.ground_truth, predictions))


class MyImage:
    def __init__(self, image_path):
        self.path = image_path
        self.name = image_path.split('/')[-1].split('.')[0]
        self.matrix = Image.open(image_path)
        self.key_points, self.descriptors = cv2.SIFT_create().detectAndCompute(
            cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2GRAY), None)

    # hashable
    def __hash__(self):
        return hash(self.name)

    # comparable
    def __eq__(self, other):
        # if type is not the same, return False
        if type(self) != type(other):
            return False
        return self.name == other.name


if __name__ == '__main__':
    # image_paths = []
    # images = [MyImage(x+'.jpg') for x in image_paths]
    # for i in [0.56, 0.57, 0.58, 0.59]:
    #     print('Threshold:', i)
    #     clusterer = ImageClusterer(similarity_threshold=i)
    #     folder_path = '../data'
    #     images = clusterer.load_images_from_folder(folder_path)
    #     graph = clusterer.cluster_images(images)
    #     #
    #     tree_roots = clusterer.construct_trees_from_components()
    #     # clusterer.display_trees(tree_roots)
    #
    #     clusterer.measure_grouping()

    for i in [-1, 0.35, 0.4, 0.45, 0.5]:
        start_time = time.time()
        clusterer = ImageClusterer([20, 12, 8, 8, 10, 7, 12, 8, 7, 13, 11, 9, 8, 7], similarity_threshold=i, sift_threshold=0.6, correspondences_threshold=20)
        folder_path = '../data'
        images = clusterer.load_images_from_folder(folder_path)
        clusterer.cluster_images(images)
        tree_roots = clusterer.construct_trees_from_components()
        # clusterer.display_trees(tree_roots)
        clusterer.measure_grouping()

        # for i in [0.36, 0.37, 0.38, 0.39]:
        #     for j in [0.8]:
        #         for k in [20]:
        #             start_time = time.time()
        #             task(i, j, k)
        #             print(f"Elapsed time: {time.time() - start_time} seconds")
        # clusterer.calculate_homography_to_root()
        # for tree in tree_roots:
        #     clusterer.stitch_images_with_perspective(tree, 5)
        print(f"Elapsed time: {time.time() - start_time} seconds")

