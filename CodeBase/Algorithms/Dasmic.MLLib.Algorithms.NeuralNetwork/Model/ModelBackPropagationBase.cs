using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    public class ModelBackPropagationBase : ModelBase
    {
        public enum EnumMode
        {
            Regression,
            Classification
        }

        public EnumMode Mode;
        public double[][] WeightsHidden { get; set; }
        public double[] TargetValues;


        public ModelBackPropagationBase(double missingValue,
                        int indexTargetAttribute,
                        int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute, countAttributes)
        {
            Mode = EnumMode.Regression;
        }

        /// <summary>
        /// Pass dummy values to base, make sure to call SetValues 
        /// to pass values
        /// </summary>
        public ModelBackPropagationBase():base(0,0,0)
        {

        }

        public double GetBiasSpecifiedLayer(int idxLayer, long idxUnit)
        {
            return GetWeight(idxLayer, GetNumberOfUpstreamUnits(idxLayer) - 1, idxUnit);

        }

        public void SetBiasSpecifiedLayer(int idxLayer, long idxUnit, double value)
        {
            SetWeight(idxLayer, GetNumberOfUpstreamUnits(idxLayer) - 1, idxUnit, value);
        }

        public long GetOutputUnitCount()
        {
            return GetNumberOfUnits(GetNumberOfLayers() - 1);
        }

        //public override abstract double RunModelForSingleData(double[] data);

        /// <summary>
        /// Will return single value which is the maximum value
        /// among all output units
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public override double RunModelForSingleData(double[] data)
        {
            double[] computedValue = GetOutputValues(data);

            //Find max Idx in computedValue
            int maxIdx = computedValue.Select((item, indx) =>
                new { Item = item, Index = indx }).
                OrderByDescending(x => x.Item).Select(x => x.Index).First();

            if (Mode == EnumMode.Classification)
            {
                return TargetValues[maxIdx];
            }
            else
                return computedValue[maxIdx];
        }

        /// <summary>
        /// Returns an array with output values of 
        /// all computed units
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public double[] GetOutputValues(double[] data)
        {
            VerifyDataForRun(data);
            double[][] data2D = Convert1Dto2D(data);
            ChangeInputLayerData(data2D);

            return GetOutput(0);
        }
    }
}
