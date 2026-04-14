#include <iostream>

class Dual {
    public: 
        double val; // value  
        double d;   // derivative 

        Dual(double v, double der = 0.0): val(v), d(der) {}

        // addition 
        Dual operator+(const Dual& other) const {
            return Dual(val + other.val, d + other.d);
        }

        // multiplication
        Dual operator*(const Dual& other) const {
            return Dual(val * other.val, d * other.val + val * other.d);
        }
        // division 
        Dual operator/(const Dual& other) const {
            return Dual(val / other.val, (d * other.val - val * other.d)/ other.val * other.val);
        }
        // subtraction 
        Dual operator-(const Dual& other) const {
            return Dual(val - other.val, d - other.d);
        }
        


};

// sin function 
Dual sin(const Dual& x) {
    return Dual(std::sin(x.val), std::cos(x.val) * x.d);
}

Dual compute(Dual x, Dual y) {
    return x * (x + y) + y * y;
}

int main() {
    int a; 
    
    // Dual x(2,1);  // derivative wrt x
    // Dual y(3,0);

    // Dual z = x * (x + y) + y * y;

   
    Dual x(2, 1); // derv wrt x 
    Dual y(3, 0);

    Dual z = (x*x + y) / (x - y); 

     std::cout << "z = " << z.val << std::endl;
    std::cout << "dz/dx = " << z.d << std::endl;



    std::cin >> a; 
}
