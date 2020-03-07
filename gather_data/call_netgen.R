# Rscript call_netgen.R point_lower point_upper cluster_num ins_num seed
library("netgen")
library("ggplot2")
args <- commandArgs(TRUE)
points.lower <- as.integer(args[1])
points.upper <- as.integer(args[2])
clu.num <- as.integer(args[3])
ins.num <- as.integer(args[4])
seed <- as.integer(args[5])

set.seed(seed)

for (i in 0:ins.num)
{
    points.num <- sample(points.lower:points.upper, 1, replace=TRUE)
    x = generateClusteredNetwork(n.points = points.num, n.cluster = clu.num, out.of.bounds.handling = "mirror", upper=100)
    x = rescaleNetwork(x, method = "global2")
    x$coordinates = x$coordinates * 1000000
    x$coordinates = round(x$coordinates, 0)
    x$lower = 0
    x$upper = 1000000
    name = sprintf("../data/TSP/cl/%d_%d.tsp", clu.num, i)
    exportToTSPlibFormat(x, name, use.extended.format=FALSE)
}