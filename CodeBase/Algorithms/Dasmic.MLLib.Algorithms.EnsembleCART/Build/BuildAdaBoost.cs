using Dasmic.MLLib.Algorithms.DecisionTree;
using Dasmic.Portable.Core;
using System.Threading.Tasks;
using System.Collections.Concurrent;


namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public class BuildAdaBoost:BuildBase
    {
        private long _depthTree;
        private double _epsilon; //In mathematics (particularly calculus), an arbitrarily small positive quantity is commonly denoted as epsilon

        public BuildAdaBoost()
        {            
            _numberOfTrees = 5;
            _epsilon = .0001; //Added for stability in division
        }


        /// <summary>
        /// 0 - Number of Trees;default 5
        /// 1 - depth of each tree;default = 1, representing a decision stump
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
                    _depthTree = (int)values[1];
        }

        public override Common.MLCore.ModelBase BuildModel(
                             double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            double[] weight = new double[trainingData[0].Length];
            int[] error = new int[trainingData[0].Length];
            //Initialize weights to 1/N
            double value = 1.0 / (double)weight.Length;
            Parallel.For(0, weight.Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, idx =>
            { 
                weight[idx] = value;    
            });

            ModelAdaBoost model =
                            new ModelAdaBoost(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1,
                                                _numberOfTrees);
            //Split the data for each tree
            //Parallelize this
            Parallel.For(0, _numberOfTrees, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, ii =>
            //for (int ii=0;ii<_numberOfTrees;ii++)
            {
                //For test only             
                BuildCARTBoost buildCart = new BuildCARTBoost();
                ModelCART modelCart = (ModelCART)buildCart.BuildModel(trainingData,
                                    attributeHeaders,
                                    indexTargetAttribute,
                                    weight);
                double[] data;
                double sumWeights=0, sumWeightedError=0;
                
                //Compute Error and Sum of Weights
                for (int rowIdx = 0; 
                    rowIdx < _trainingData[0].Length; 
                    rowIdx++)
                {
                    
                    data = GetLinearArray(_trainingData, rowIdx, _trainingData.Length - 2);
                    value = modelCart.RunModelForSingleData(data);
                    if (SupportFunctions.DoubleCompare(value, 
                            _trainingData[_indexTargetAttribute][rowIdx]))
                        error[rowIdx] = 0;
                    else
                        error[rowIdx] = 1;

                    //Compute data for mis-classification rate
                    //This has to be done after error computation
                    sumWeights += weight[rowIdx];
                    sumWeightedError += weight[rowIdx] * error[rowIdx];
                }

                double misClassificationRate = (sumWeightedError+_epsilon) / (sumWeights + _epsilon);
                
                //Diagnostics Messages
                System.Diagnostics.Debug.WriteLine("\n\nSumWeights:"
                            + sumWeights.ToString());
                System.Diagnostics.Debug.WriteLine("Sum Weighted Error:"
                            + sumWeightedError.ToString());
                System.Diagnostics.Debug.WriteLine("MisClassification Rate:"
                            +   misClassificationRate.ToString());
                
                //Compute Stage
                double stage = System.Math.Log(
                                (1.0 - misClassificationRate) 
                                / misClassificationRate);

                System.Diagnostics.Debug.WriteLine("Stage:"
                            + stage.ToString());
                
                model.AddTree(stage,ii,modelCart);
                
                //Update the Weights
                for (int rowIdx = 0;
                    rowIdx < _trainingData[0].Length;
                    rowIdx++)
                {
                    weight[rowIdx] = weight[rowIdx] *
                                System.Math.Pow(System.Math.E, 
                                stage * error[rowIdx]);
                }

            }); //Number of trees

         
            return model;
        }
    }
}
