using Dasmic.MLLib.Algorithms.DecisionTree;
using Dasmic.MLLib.Math.Statistics;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public class ModelAdaBoost: ModelBase
    {
        protected List<double> _allStages { get; private set; }
        
        public ModelAdaBoost(double missingValue,
                          int indexTargetAttribute, 
                          int countAttributes, long noOfTrees) :
                                base(missingValue, indexTargetAttribute, countAttributes,noOfTrees)
        {
            _allStages = new List<double>();
        }

        public void AddTree(double stage, 
            long idxTree, ModelCART tree)
        {
            //Cannot use a Dictionary here as stage can be duplicated
            //Hence have to use this approach
            base.AddTree(idxTree, tree);            
            _allStages.Add(stage);
        }

        public override
            double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double finalValue = 0;
            double[] values = new double[_allTrees.Length];
            //Iterate through all trees
            object monitor = new object();
            Parallel.For(0, _allTrees.Length, 
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idx =>
            //for(int idx=0;idx<allTrees.Count;idx++)
            {
                ModelCART modelCart = _allTrees[idx];
                values[idx] =
                             _allStages[idx] *
                             modelCart.RunModelForSingleData(data);
                lock (monitor)
                    finalValue += values[idx];
            });
            

            //double finalValue = Dispersion.Mode(values);
            return finalValue;
        }
    }
}
