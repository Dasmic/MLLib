using System;
using Dasmic.MLLib.Common.Exceptions;
using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    //Methods are not declared static since they can be inherited and overridden
    public class DecisionTreePruning
    {

        protected DecisionTreeNode PruneTree(
            double[][] trainingData,
            int indexTargetAttribute,
            DecisionTreeNode root)
        {
            return ErrorBasedPruning(trainingData,
                            indexTargetAttribute,
                            root);
        }

        public int getNumberMisclassified(
            double[][] trainingData,
            int indexTargetAttribute,
            ModelBase mb)
        {
            int colIdx;
            List<double> tmpList;

            double value;
            
            if (trainingData == null) throw new InvalidTrainingDataException();
            int misClassify = 0;
            for (int rowIdx = 0; rowIdx < 
                trainingData[0].Length; rowIdx++)
            {
                //Transfer to temp array
                tmpList = new List<double>();
                for (colIdx = 0; colIdx < trainingData.Length; colIdx++)
                {
                    if (colIdx != indexTargetAttribute)
                        tmpList.Add(trainingData[colIdx][rowIdx]);
                }
                value = mb.RunModelForSingleData(tmpList.ToArray());
                if (value == Constants.MISSING_VALUE)
                    misClassify++;//Count missing vsalue as misclassified
                else if (value - trainingData[indexTargetAttribute][rowIdx] >
                        System.Math.Abs(Constants.VALUE_THRESHOLD))
                            misClassify++;

            }

            return misClassify;
        }


        public float getMisclassificationRate(
            double[][] trainingData,
            int indexTargetAttribute,
            ModelBase modelBase)
        {
            int numberMisclassified =
                getNumberMisclassified(trainingData,
                                        indexTargetAttribute, modelBase);

            return (float)numberMisclassified /
                        (float)trainingData[0].Length;
        }
           


        //Error Based Pruning
        protected DecisionTreeNode 
            ErrorBasedPruning(
            double[][] trainingData,
            int indexTargetAttribute,
            DecisionTreeNode root)
        {
            return null;
        }
                             
    }

}
