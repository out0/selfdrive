#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

struct CompareByFloat
{
    bool operator()(const std::tuple<int, int, float> &a,
                     const std::tuple<int, int, float> &b) const
    {
        // std::priority_queue is a max-heap by default;
        // using > here gives you a min-heap (smallest float on top)
        return std::get<2>(a) > std::get<2>(b);
    }
};

void printTop(const std::tuple<int, int, float> &t)
{
    std::cout << "top element: ("
              << std::get<0>(t) << ", "
              << std::get<1>(t) << ", "
              << std::get<2>(t) << ")" << std::endl;
}

int main()
{
    // Declaring a min-heap (by float value)
    std::priority_queue<std::tuple<int, int, float>,
                         std::vector<std::tuple<int, int, float>>,
                         CompareByFloat>
        min_queue;

    // Insert elements - O(log n)
    min_queue.push({30, 10, 3.1f});
    min_queue.push({50, 10, 4.1f});
    min_queue.push({10, 10, 1.1f});
    min_queue.push({40, 10, 5.1f});
    min_queue.push({20, 10, 2.1f});

    // Access top element (smallest float) - O(1)
    printTop(min_queue.top()); // (10, 10, 1.1)

    // Remove top element - O(log n)
    min_queue.pop();

    printTop(min_queue.top()); // (20, 10, 2.1)
}