using System;
using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.Regression
{
    public class ModelLogisticBase : ModelBase
    {

        public double B0;
        public List<double> B;

        public ModelLogisticBase(double missingValue,
                                int indexTargetAttribute, int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute,countAttributes)
        {
            B0 = missingValue;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data">Can we have n-dimension array</param>
        /// <param name="targetAttributeIndex"></param>
        /// <returns></returns>
        public override
            double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            if (data.Length < 2)
                throw new InvalidTrainingDataException();
            if (B0 == _missingValue)
                throw new ModelNotReadyException();

            double pow = 0;
            for (int col = 0; col < B.Count; col++)
            {
                if (col != _origTargetAttributeIndex)
                {
                    pow += B[col] * data[col]; //Single row of attributes
                }
            }
            pow += B0;
            return 1 / (1 + System.Math.Exp(-pow));
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
