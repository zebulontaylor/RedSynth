module \$_NOT_ (input A, output Y); NOT _TECHMAP_REPLACE_ (.A(A), .Y(Y)); endmodule
module \$_AND_ (input A, B, output Y); AND _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y)); endmodule
module \$_OR_ (input A, B, output Y); OR _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y)); endmodule
module \$_XOR_ (input A, B, output Y); XOR _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y)); endmodule

module \$_MUX_ (input A, B, S, output Y);
    MUX _TECHMAP_REPLACE_ (.A(A), .B(B), .S(S), .Y(Y));
endmodule
