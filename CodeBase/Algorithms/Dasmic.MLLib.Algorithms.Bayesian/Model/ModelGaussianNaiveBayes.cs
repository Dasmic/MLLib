using System;
using System.Collections.Generic;
using Dasmic.MLLib.Math.Statistics;

namespace Dasmic.MLLib.Algorithms.Bayesian
{
    public class ModelGaussianNaiveBayes:ModelBase
    {
        internal List<double[][]> ClassMeanMatrix;
        internal List<double[][]> ClassSDMatrix;
        internal double[] targetValues;

        public ModelGaussianNaiveBayes(double missingValue,
                               int indexTargetAttribute,
                               int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute, 
                                    countAttributes)
        {
            
        }

        //Will work for discrete and continuous values
        public override double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double result = _missingValue;
            double maxProb = Double.NegativeInfinity;

            DistributionNormal dn = new DistributionNormal();

            //Do for each target attribute
            for(int idx=0;idx<ClassMeanMatrix.Count;idx++)
            {
                double prob = 1;
                for (int col = 0; col < data.Length; col++)
                {
                    //Multiple probability of each attribute
                    prob = prob * dn.ProbabilityDensityFunction(data[col],
                                            ClassMeanMatrix[idx][col][0],
                                            ClassSDMatrix[idx][col][0]);
                }
                
                if (prob > maxProb)
                {
                    maxProb = prob;
                    result = targetValues[idx]; //It is important that this index is same as classMeanMatrix and classSDMatrix 
                }
            }
            return result;
        }
    }
}
