#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <unordered_set>

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad = 0.0;

    std::vector<std::shared_ptr<Value>> prev;
    std::function<void()> backward;

    Value(double v) : data(v) {}



    // 🔹 Backward pass (topological order)
    void backward_pass() {
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<Value*> visited;

        // DFS to build topological order
        std::function<void(std::shared_ptr<Value>)> build =
            [&](std::shared_ptr<Value> v) {
                if (!visited.count(v.get())) {
                    visited.insert(v.get());
                    for (auto& p : v->prev)
                        build(p);
                    topo.push_back(v);
                }
            };

        build(shared_from_this());

        // Initialize output gradient
        this->grad = 1.0;

        // Traverse in reverse
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward)
                (*it)->backward();
        }
    }
};



std::shared_ptr<Value> operator+(std::shared_ptr<Value> a,
                                 std::shared_ptr<Value> b) {
    auto out = std::make_shared<Value>(a->data + b->data);
    out->prev = {a, b};

    out->backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };

    return out;
}


std::shared_ptr<Value> operator*(std::shared_ptr<Value> a,
                                 std::shared_ptr<Value> b) {
    auto out = std::make_shared<Value>(a->data * b->data);
    out->prev = {a, b};

    out->backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };

    return out;
}

std::shared_ptr<Value> sin(std::shared_ptr<Value> x) {
    auto out = std::make_shared<Value>(std::sin(x->data));
    out->prev = {x};

    out->backward = [x, out]() {
        x->grad += std::cos(x->data) * out->grad;
    };

    return out;
}


int main() {
    auto x1 = std::make_shared<Value>(2.0);
    auto x2 = std::make_shared<Value>(3.0);

    auto y = x1 * (x1 + x2) + x2 * x2;

    y->backward_pass();

    std::cout << "dy/dx1 = " << x1->grad << std::endl;
    std::cout << "dy/dx2 = " << x2->grad << std::endl;
}