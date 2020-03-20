library("tspmeta")
options=list("RUE", "cl", "explosion", "implosion", "cluster","compression","expansion","grid","linearprojection", "rotation")

fileConn<-file("tspmeta_feature.txt", 'w+')
write('angle_min,angle_median,angle_mean,angle_max,angle_sd,angle_span,angle_coef_of_var,centroid_centroid_x,centroid_centroid_y,centroid_dist_min,centroid_dist_median,centroid_dist_mean,centroid_dist_max,centroid_dist_sd,centroid_dist_span,centroid_dist_coef_of_var,cluster_01pct_number_of_clusters,cluster_01pct_mean_distance_to_centroid,cluster_05pct_number_of_clusters,cluster_05pct_mean_distance_to_centroid,cluster_10pct_number_of_clusters,cluster_10pct_mean_distance_to_centroid,bounding_box_10_ratio_of_cities_outside_box,bounding_box_20_ratio_of_cities_outside_box,bounding_box_30_ratio_of_cities_outside_box,chull_area,chull_points_on_hull,distance_distances_shorter_mean_distance,distance_distinct_distances,distance_mode_frequency,distance_mode_quantity,distance_mode_mean,distance_mean_tour_length,distance_sum_of_lowest_edge_values,distance_min,distance_median,distance_mean,distance_max,distance_sd,distance_span,distance_coef_of_var,modes_number,mst_depth_min,mst_depth_median,mst_depth_mean,mst_depth_max,mst_depth_sd,mst_depth_span,mst_depth_coef_of_var,mst_dists_min,mst_dists_median,mst_dists_mean,mst_dists_max,mst_dists_sd,mst_dists_span,mst_dists_coef_of_var,mst_dists_sum,nnds_min,nnds_median,nnds_mean,nnds_max,nnds_sd,nnds_span,nnds_coef_of_var', fileConn)


for (option in options)
{
    for (i in 1:1000)
    {
        instance=sprintf("data/TSP/%s/%d.tsp", option, i)
        x=read_tsplib_instance(instance)
        t = system.time({y=features(x)})
        line = instance
        for (feature in y)
        {
            line = sprintf('%s,%f', line, feature)
        }
        line = sprintf('%s,%f', line, t[3])
        write(line, fileConn)
    }
}

close(fileConn)
