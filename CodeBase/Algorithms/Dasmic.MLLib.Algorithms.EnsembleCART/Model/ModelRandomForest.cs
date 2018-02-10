using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Algorithms.DecisionTree;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public class ModelRandomForest : ModelBase
    {
        public ModelRandomForest(double missingValue,
                   int indexTargetAttribute, int countAttributes, long noOfTrees) :
                                base(missingValue, 
                                    indexTargetAttribute, countAttributes,noOfTrees)
        {

        }

        public void AddTree(long idxTree, ModelCART tree, long[] index)
        {
            _allTrees[idxTree]=tree;
        }

    }
}
