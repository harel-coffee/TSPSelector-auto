# Rscript call_tspgen.R operator point_lower point_upper ins_num seed

library("netgen")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/utilities.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.explosion.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.implosion.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.cluster.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.compression.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.expansion.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.grid.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.linearprojection.R")
source("../../ACPP/instance_set/diverse_mutator/tspgen-master/R/mutator.rotation.R")

args <- commandArgs(TRUE)
operator <- toString(args[1])
points.lower <- as.integer(args[2])
points.upper <- as.integer(args[3])
ins.num <- as.integer(args[4])
seed <- as.integer(args[5])

set.seed(seed)

for (i in 1:ins.num)
{
    points.num <- sample(points.lower:points.upper, 1, replace=TRUE)
    x = generateRandomNetwork(n.points = points.num, lower = 0, upper = 1)

    if (operator == "explosion")
    {
        x$coordinates = doExplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "implosion")
    {
        x$coordinates = doImplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "cluster")
    {
        x$coordinates = doClusterMutation(x$coordinates, pm=0.4)
    }
    if (operator == "compression")
    {
        x$coordinates = doCompressionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "expansion")
    {
        x$coordinates = doExpansionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "grid")
    {
        x$coordinates = doGridMutation(x$coordinates, box.min=0.3, box.max=0.3, p.rot=0, p.jitter=0, jitter.sd=0.05)
    }
    if (operator == "linearprojection")
    {
        x$coordinates = doExplosionMutation(x$coordinates, pm=0.4, p.jitter=0, jitter.sd=0.05)
    }
    if (operator == "rotation")
    {
        x$coordinates = doRotationMutation(x$coordinates, pm=0.4)
    }

    x = rescaleNetwork(x, method = "global2")
    x$coordinates = x$coordinates * 1000000
    x$coordinates = round(x$coordinates, 0)
    x$coords = relocateDuplicates(x$coords)
    x$lower = 0
    x$upper = 1000000

    name = sprintf("../data/TSP/%s/%d.tsp", operator, i)
    exportToTSPlibFormat(x, name, use.extended.format=FALSE)
}
