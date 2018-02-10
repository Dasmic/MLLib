using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Math.Statistics;

namespace Dasmic.MLLib.Algorithms.Bayesian
{
    public class BuildGaussianNaiveBayes:BuildBase
    {
       
        public override Common.MLCore.ModelBase 
                BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            double rowCount = trainingData[0].Length;

            ModelGaussianNaiveBayes model = new ModelGaussianNaiveBayes(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            //Get unique value in Target Attribute
            HashSet<double> uniqueTargetValues =
                    GetUniqueValues(trainingData[indexTargetAttribute]);
            
            model.targetValues = new double[uniqueTargetValues.Count];

            //Split input on classes and store targetValues
            List<double[][]> classInputMatrix =
                            GetClassBasedInputMatrix(uniqueTargetValues,
                                ref model.targetValues);

            model.ClassMeanMatrix =
                            GetClassMeanMatrix(classInputMatrix);

            model.ClassSDMatrix =
                            GetClassStandardDeviationMatrix(classInputMatrix,model.ClassMeanMatrix);

            return model;
        }
    }
}
