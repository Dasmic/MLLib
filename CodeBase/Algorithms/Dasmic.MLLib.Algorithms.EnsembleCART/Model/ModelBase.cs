using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Algorithms.DecisionTree;
using Dasmic.MLLib.Math.Statistics;
using System.Collections.Generic;
using System.Collections.Concurrent;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public abstract class ModelBase : MLLib.Common.MLCore.ModelBase
    {
        protected ModelCART[] _allTrees { get; private set;}

        public ModelBase(double missingValue,
                                int indexTargetAttribute, int countAttributes,
                                long noOfTrees) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {
            _allTrees = new ModelCART[noOfTrees];
        }

        public virtual void AddTree(long idxTree,
                                        ModelCART tree)
        {
            _allTrees[idxTree]=tree;
        }

        public override
            double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double[] values = 
                        new double[_allTrees.Length];
            int idx = 0;

            //Iterate through all trees
            foreach(ModelCART modelCart in _allTrees)
            {
                values[idx++]=
                            modelCart.RunModelForSingleData(data);
            }
            
            double finalValue = Dispersion.Mode(values);
            return finalValue;
        }


        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }

    }
}
