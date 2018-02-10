 using Dasmic.MLLib.Math.Statistics;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.Regression
{
    public class BuildLinearSimple:BuildBase
    {
        

        //Training Data Should have only Two Attributes
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            VerifyData(trainingData,attributeHeaders,indexTargetAttribute);
            
            double[] x, y;
            ModelLinearBase model = new ModelLinearBase(_missingValue,
                                        _indexTargetAttribute, _trainingData.Length - 1);

            //Additional Checks
            if (trainingData.Length != 2)
                throw new InvalidTrainingDataException();
            if (indexTargetAttribute == 0)
            {
                x = trainingData[1];
                y = trainingData[0];
            }
            else
            {
                x = trainingData[0];
                y = trainingData[1];
            }

            //x,y are arrays
            model.B1 = Dispersion.CorrelationPearson(x, y)
                        * (Dispersion.StandardDeviationSample(y) /
                                Dispersion.StandardDeviationSample(x));

            model.B0 = Dispersion.Mean(y) -
                model.B1 * Dispersion.Mean(x);

            return model;
        }

    }
}

