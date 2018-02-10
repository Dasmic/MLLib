using Dasmic.MLLib.Algorithms.DecisionTree;
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public class BuildRandomForest : BuildBase
    {
        private int _numberOfFeaturesPerTree;

        public BuildRandomForest()
        {
            _numberOfFeaturesPerTree = int.MaxValue;
            _numberOfTrees = 5;
        }

        /// <summary>
        /// 0 - Number of Trees;default 5
        /// 1 - Number of features per Tree;default = sqrt(number of features)
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
                    _numberOfFeaturesPerTree = (int)values[1];
        }


        public override Common.MLCore.ModelBase BuildModel(
                             double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelRandomForest model =
                            new ModelRandomForest(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1, 
                                                _numberOfTrees);

            //By default samples/tree is same as original samples
            if (_numberOfFeaturesPerTree == int.MaxValue)
                _numberOfFeaturesPerTree = (int)System.Math.Ceiling(
                                            System.Math.Sqrt((double)_trainingData.Length));

            //Create for each 
            //Parallelize this
            Parallel.For(0, _numberOfTrees, 
                new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },ii =>
            //for (int ii = 0; ii < _numberOfTrees; ii++)
            {             
                //For test only             
                BuildCART buildCart = new BuildCART();
                buildCart.SetParametersForRandomForest(_numberOfFeaturesPerTree);
                ModelCART modelCart = (ModelCART)buildCart.BuildModel(trainingData,
                                    attributeHeaders,
                                    indexTargetAttribute);
                model.AddTree(ii,modelCart);
            });//Number of trees

            return model;
        } 
    }
}
