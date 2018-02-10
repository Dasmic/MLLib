using Dasmic.MLLib.Algorithms.DecisionTree;
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    /// <summary>
    /// Build Bootstrapped Aggregation (Bagging) Decision Tree
    /// </summary>
    public class BuildBaggedDecisionTree : BuildBase
    {
        private long _numberOfSamplesPerTree;

        public BuildBaggedDecisionTree()
        {
            _numberOfSamplesPerTree = long.MaxValue;
            _numberOfTrees = 5;
        }

        /// <summary>
        /// 0 - Number of Trees;default 5
        /// 1 - Number of training samples per Tree;default = number of training sample
        /// </summary>
        /// <param name="values"></param>
        public override void
            SetParameters(params double[] values)
        {
            if (values.Length > 0)
                if (values[0] != double.NaN)
                    _numberOfTrees = (int)values[0];
            if (values.Length > 1)
                if (values[0] != double.NaN)
                    _numberOfSamplesPerTree = (int)values[1];        
        }


        public override Common.MLCore.ModelBase BuildModel(
                             double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelBaggedDecisionTree model =
                            new ModelBaggedDecisionTree(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1,
                                                _numberOfTrees);

            //By default samples/tree is same as original samples
            if (_numberOfSamplesPerTree == long.MaxValue)
                _numberOfSamplesPerTree = _trainingData[0].Length;

            //Split the data for each tree
            //Parallelize this
            Parallel.For(0, _numberOfTrees, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },ii =>
            //for (int ii=0;ii<_numberOfTrees;ii++)
            {
                Random rnd = new Random();
                ConcurrentBag<long> trainingDataRowIndices =
                    new ConcurrentBag<long>();

                //Initialize the rows                
                for (long idx = 0;
                                idx < _numberOfSamplesPerTree; idx++)
                {
                    //Random Sampling with Replacement
                    long rowIdx = rnd.Next(0, _trainingData[0].Length - 1);

                    //rowIdx = idx + startIdx;
                    //rowIdx = rowIdx > _noOfDataSamples - 1 ?
                    //                    rowIdx - _noOfDataSamples : rowIdx;
                    trainingDataRowIndices.Add(rowIdx);
                }

                //For test only             
                BuildCART buildCart = new BuildCART();
                ModelCART modelCart = (ModelCART)buildCart.BuildModel(trainingData,
                                    attributeHeaders,
                                    indexTargetAttribute,
                                    trainingDataRowIndices);
                model.AddTree(ii,modelCart);
            }); //Number of trees
            
            return model;
        }
    }
}
