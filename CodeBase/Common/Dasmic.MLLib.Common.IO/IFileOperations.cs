
namespace Dasmic.MLLib.Common.IO
{
    public interface IFileOperations
    {
        void Write(FileData fd, string fullFilePath);
        void Write(double[][] values,
                        string[] attributeHeaders,
                        string fullFilePath);
        FileData Read(string fullFilePath, int maxParallelThreads);
    }
}
