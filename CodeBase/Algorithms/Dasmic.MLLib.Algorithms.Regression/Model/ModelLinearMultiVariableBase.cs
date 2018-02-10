using System;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.Regression
{
    public class ModelLinearMultiVariableBase: ModelBase
    {
        private double[] _B; //Array of coefficients

        public ModelLinearMultiVariableBase(double missingValue,
                              int indexTargetAttribute, int countAttributes) :
                                base(missingValue, indexTargetAttribute,countAttributes)
        {
            _B = new double[countAttributes + 1];
        }

        public void SetCoeff(int index, double value)
        {
            _B[index] = value;
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
            double value=0;
            //Note intercept is at last index
            for(int idx=0;idx<data.Length;idx++)
            {
                value += _B[idx] * data[idx];
            }
            value += _B[data.Length] * 1.0;

            return value;          
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
