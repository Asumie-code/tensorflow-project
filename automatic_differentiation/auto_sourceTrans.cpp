
#include <vector>
#include <iostream>

enum class OpType
{
    Multiply,
    Sin,
    Add
};

struct op
{
    OpType type;
    int lhs;
    int rhs;
    int result;
};

std::vector<op> ops = {
    {OpType::Multiply, 1, 2, 3},
    {OpType::Sin, 1, 0, 4},
    {OpType::Add, 3, 4, 5}};

int main()
{
    for (auto it = ops.rbegin(); it != ops.rend(); ++it)
    {
        switch (it->type)
        {
        case OpType::Add:
            std::cout
                << "a" << it->lhs
                << " += a" << it->result << ";\n";

            std::cout
                << "a" << it->rhs
                << " += a" << it->result << ";\n";
            break;

        case OpType::Multiply:
            std::cout
                << "a" << it->lhs
                << " += a" << it->result
                << " * w" << it->rhs << ";\n";

            std::cout
                << "a" << it->rhs
                << " += a" << it->result
                << " * w" << it->lhs << ";\n";
            break;

        case OpType::Sin:
            std::cout
                << "a" << it->lhs
                << " += a" << it->result
                << " * cos(w" << it->lhs << ");\n";
            break;
        }
    }
}
