 namespace Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction
{
    public interface IPoolingFunction
    {
        double GetValue();
        void AddValue(double value);
    }
}
