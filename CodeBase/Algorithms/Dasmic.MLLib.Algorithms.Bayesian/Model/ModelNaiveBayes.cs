using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.Bayesian
{
    public class ModelNaiveBayes:ModelBase
    {
        //key is unique value of TargetClass
        internal ConcurrentDictionary<double, double> ProbTargetClass;

        //key is unique value of TargetClass, while for
        //second dictionary Key is unique value of each attribute
        internal ConcurrentDictionary<int, ConcurrentDictionary<double,
            ConcurrentDictionary<double, double>>> ProbAllClasses;

        public ModelNaiveBayes(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute, 
                                    countAttributes)
        {
            ProbTargetClass = new ConcurrentDictionary<double, double>();
            ProbAllClasses = new ConcurrentDictionary<int, 
                ConcurrentDictionary<double, ConcurrentDictionary<double, double>>>();
        }

        //Will work for discrete values only
        public override double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            double result = _missingValue;
            double maxProb = Double.NegativeInfinity;
            //Do for each target attribute

            foreach (double value in ProbTargetClass.Keys)    
            {
                double prob = 1;
                //Compute probability for each data col
                //based on apriori probability
                for (int col = 0; col < data.Length; col++)//Do not Parallelize
                {
                    prob = prob * ((ProbAllClasses[col])[value])[data[col]];
                }
                prob = prob * ProbTargetClass[value];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    result = value;
                }
            }
       
            return result;
        }
    }
}
