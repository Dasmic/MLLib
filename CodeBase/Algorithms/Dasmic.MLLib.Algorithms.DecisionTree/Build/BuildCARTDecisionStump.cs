using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    /// <summary>
    /// CART optimized for AdaBoost
    /// </summary>
    public class BuildCARTBoost:BuildCART
    {
        protected double [] _trainingDataWeights;
        protected long _depthTree;

        public BuildCARTBoost()
        {
            _depthTree = 1;
        }


        /// <summary>
        /// Default value is 1 (for Decision Stump)
        /// </summary>
        /// <param name="value"></param>
        public void SetTreeDepth(int value)
        {
            _depthTree = value; //For decision tree stump
        }


        public Common.MLCore.ModelBase BuildModel(
                         double[][] trainingData,
                         string[] attributeHeaders,
                         int indexTargetAttribute,
                         double[] trainingDataWeights)
        {
            ConcurrentBag<long> trainingDataRowIndices =
                        GetTrainingDataForAllRows(trainingData);

            _trainingDataWeights = trainingDataWeights;

            return BuildModel(trainingData,
                            attributeHeaders,
                            indexTargetAttribute,
                            trainingDataRowIndices, false);
        }

        /// <summary>
        /// Use Weights when computing gini impurity
        /// </summary>
        /// <param name="data"></param>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        protected override double GetGiniImpurity(double[][] data,
                                            int row,
                                            int col)
        {
            //Gini values are selected based on weights
            //More the weights, lower the value
            //Lower gini values are selected
            return GetGiniImpurity(col, data[col][row])/
                                _trainingDataWeights[row];
        }

        //Any extra stopping condition, primarily added for boosting
        //which needs decision stumps
        protected override bool GetAdditionalStoppingCondition(DecisionTreeNode dtn)
        {
            //Find depth
            int depth=0;
            while (dtn.Parent != null)
                depth++;

            if (depth >= _depthTree) return true;
            return false;
        }


    }
}
