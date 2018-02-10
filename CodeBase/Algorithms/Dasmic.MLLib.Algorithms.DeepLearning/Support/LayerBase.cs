using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public abstract class LayerBase
    {
        protected LayerBase _upstreamLayer;
        protected int _maxParallelThreads;        

        protected UnitBase[] FilterUnits;

        #region Properties
        public int MaxParallelThreads { get; set; }
        public double WeightBaseValue { get; set; }
        #endregion Properties


        public LayerBase()
        {
            MaxParallelThreads = -1;
            WeightBaseValue = .005; //default
        }
        

        /// <summary>
        /// Returns number of Filter Units
        /// 
        /// NOTE: Does not incluse bias term
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfFilterUnits()
        {
            return FilterUnits.Length;
        }

        public UnitBase GetFilterUnit(int idx)
        {
            return FilterUnits[idx];
        }

        /// <summary>
        /// Does not include bias term
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfUpstreamUnits()
        {
            return _upstreamLayer.GetNumberOfFilterUnits();
        }

        public int GetValueMapNoOfColumns()
        {
            return FilterUnits[0].GetValueMapNoOfColumns();
        }
          
        public int GetValueMapNoOfRows()
        {
            return FilterUnits[0].GetValueMapNoOfRows();
        }
        
    }
}
