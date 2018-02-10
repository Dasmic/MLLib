using System;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.Regression
{
    public class ModelLinearBase:ModelBase
    {
        public double B0;
        public double B1;

        public ModelLinearBase(double missingValue,
                                int indexTargetAttribute, int countAttributes) :
                                base(missingValue, indexTargetAttribute,countAttributes)
        {
            B0 = missingValue;
            B1 = missingValue;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data">should have only one value/param>
        /// <param name="targetAttributeIndex"></param>
        /// <returns></returns>
        public override
            double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);

            if (data.Length > 1)
                throw new InvalidDataException();
            if (B0 == _missingValue)
                throw new ModelNotReadyException();

            double x = data[0];
            
            return B0 + B1 * x;
        }

        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }
    }
}
