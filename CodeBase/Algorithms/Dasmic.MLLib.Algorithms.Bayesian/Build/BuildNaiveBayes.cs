using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.Bayesian
{
    public class BuildNaiveBayes:BuildBase
    {
        public override Common.MLCore.ModelBase 
                BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            double rowCount = trainingData[0].Length;

            ModelNaiveBayes model = new ModelNaiveBayes(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            //Get unique value in Target Attribute
            HashSet<double> uniqueTargetValues =
                    GetUniqueValues(trainingData[indexTargetAttribute]);
            
   
            double[] targetValues = new double[uniqueTargetValues.Count];

            //Split input on classes
            List<double[][]> classInputMatrix =
                            GetClassBasedInputMatrix(uniqueTargetValues,
                                ref targetValues);

            //Add in probabilities for TargetClass
            //for (int idx=0;idx<targetValues.Length;idx++)
            Parallel.For(0, targetValues.Length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, idx =>
            {
                double val = classInputMatrix[idx][0].Length / rowCount;
                model.ProbTargetClass.AddOrUpdate(
                    targetValues[idx], val, ((key, oldValue) => oldValue));
            });

            //Get unique values in each class
            //Assume last attribute is target

            //for (int col = 0; col < _trainingData.Length-1; col++)
            Parallel.For(0, _trainingData.Length - 1, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, col =>
            {
                ConcurrentDictionary<double, ConcurrentDictionary<double, double>> targetValueBasedProb =
                            new ConcurrentDictionary<double, ConcurrentDictionary<double, double>>();

                HashSet<double> uniqueValues = GetUniqueValues(trainingData[col]);
                    //Do for each target value
                    //e.g. class = go-out
                    for (int idx = 0; idx < classInputMatrix.Count; idx++)
                {
                    ConcurrentDictionary<double, double> classBasedAttrProb =
                                            new ConcurrentDictionary<double, double>();
                        //Do for each individual value in each attribute
                        //Doing process:
                        //count(weather = sunny ^ class = go-out)/count(class = go-out)
                        foreach (double val in uniqueValues)
                    {
                            //Get count in classInputMatrix
                            double classCount =
                                classInputMatrix[idx][col].Count(v => v == val);
                        double condProb = classCount /
                                classInputMatrix[idx][col].Length;
                        classBasedAttrProb.AddOrUpdate(val, condProb, (key, oldValue)
                                             => oldValue);
                    }
                    targetValueBasedProb.AddOrUpdate(targetValues[idx], classBasedAttrProb, (key, oldValue) => oldValue);
                }
                model.ProbAllClasses.AddOrUpdate(col, targetValueBasedProb,
                        (key, oldValue) => oldValue);
            });//Parallel
            return model;
        }
    }
}
