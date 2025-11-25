(* techmap_celltype = "$add" *)
module _80_redstone_add8 (A, B, Y);
	parameter A_SIGNED = 0;
	parameter B_SIGNED = 0;
	parameter A_WIDTH = 1;
	parameter B_WIDTH = 1;
	parameter Y_WIDTH = 1;

	input [A_WIDTH-1:0] A;
	input [B_WIDTH-1:0] B;
	output [Y_WIDTH-1:0] Y;

	generate
		if (A_WIDTH == 8 && B_WIDTH == 8 && Y_WIDTH == 8) begin
			RS_ADD8 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
		end else begin
			wire _TECHMAP_FAIL_ = 1;
		end
	endgenerate
endmodule

(* techmap_celltype = "$sub" *)
module _80_redstone_sub8 (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && B_WIDTH == 8 && Y_WIDTH == 8) begin
            wire [7:0] not_b;
            wire [7:0] sum1;
            RS_NOT8 u_not (.A(B), .Y(not_b));
            RS_ADD8 u_add1 (.A(A), .B(not_b), .Y(sum1));
            RS_ADD8 u_add2 (.A(sum1), .B(8'd1), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$shl" *)
module _80_redstone_shl8 (A, B, Y);
	parameter A_SIGNED = 0;
	parameter B_SIGNED = 0;
	parameter A_WIDTH = 1;
	parameter B_WIDTH = 1;
	parameter Y_WIDTH = 1;

	input [A_WIDTH-1:0] A;
	input [B_WIDTH-1:0] B;
	output [Y_WIDTH-1:0] Y;

	generate
		if (A_WIDTH == 8 && B_WIDTH == 8 && Y_WIDTH == 8) begin
			RS_SHL8 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
		end else begin
			wire _TECHMAP_FAIL_ = 1;
		end
	endgenerate
endmodule

(* techmap_celltype = "$dff" *)
module _80_redstone_dff8 (CLK, D, Q);
	parameter WIDTH = 1;
	parameter CLK_POLARITY = 1;

	input CLK;
	input [WIDTH-1:0] D;
	output [WIDTH-1:0] Q;

	generate
		if (WIDTH == 8 && CLK_POLARITY == 1) begin
			RS_DFF8 _TECHMAP_REPLACE_ (.C(CLK), .D(D), .Q(Q));
		end else begin
			wire _TECHMAP_FAIL_ = 1;
		end
	endgenerate
endmodule

(* techmap_celltype = "$dffe" *)
module _80_redstone_dffe8 (CLK, EN, D, Q);
	parameter WIDTH = 1;
	parameter CLK_POLARITY = 1;
	parameter EN_POLARITY = 1;

	input CLK, EN;
	input [WIDTH-1:0] D;
	output [WIDTH-1:0] Q;

	generate
		if (WIDTH == 8 && CLK_POLARITY == 1 && EN_POLARITY == 1) begin
            wire [7:0] next_d;
            assign next_d = EN ? D : Q;
			RS_DFF8 _TECHMAP_REPLACE_ (.C(CLK), .D(next_d), .Q(Q));
		end else begin
			wire _TECHMAP_FAIL_ = 1;
		end
	endgenerate
endmodule

(* techmap_celltype = "$and" *)
module _80_redstone_and8 (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && B_WIDTH == 8 && Y_WIDTH == 8) begin
            RS_AND8 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$or" *)
module _80_redstone_or8 (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && B_WIDTH == 8 && Y_WIDTH == 8) begin
            RS_OR8 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$xor" *)
module _80_redstone_xor8 (A, B, Y);
    parameter A_SIGNED = 0;
    parameter B_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter B_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    input [B_WIDTH-1:0] B;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && B_WIDTH == 8 && Y_WIDTH == 8) begin
            RS_XOR8 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$not" *)
module _80_redstone_not8 (A, Y);
    parameter A_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && Y_WIDTH == 8) begin
            RS_NOT8 _TECHMAP_REPLACE_ (.A(A), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

module RS_RAM8_BRAM (CLK0, CLK1, A0, A1, D0, D1, WE0, WE1);
	input CLK0, CLK1;
	input [7:0] A0, A1;
	output [7:0] D0;
	input [7:0] D1;
	input WE0; // Read port enable usually 1 bit or unused
	input [7:0] WE1; // Write port enable 8 bits

	// Assuming Single Port usage where Read and Write happen at same address
	// or we prioritize one.
	// For RS_RAM8, we map:
	// CLK -> CLK0 (Read Clock) - assuming same clock
	// ADDR -> A0 (Read Address) - assuming A0 == A1 if WE is high
	// DI -> D1 (Write Data)
	// DO -> D0 (Read Data)
	// WE -> WE1[0] (Write Enable - assuming all bits same)

	RS_RAM8 _TECHMAP_REPLACE_ (
		.CLK(CLK0),
		.ADDR(A0),
		.DI(D1),
		.DO(D0),
		.WE(WE1[0])
	);
endmodule

module RS_ROM8_BRAM (CLK0, A0, D0);
	input CLK0;
	input [7:0] A0;
	output [7:0] D0;

	RS_ROM8 _TECHMAP_REPLACE_ (
		.CLK(CLK0),
		.ADDR(A0),
		.DO(D0)
	);
endmodule

(* techmap_celltype = "$mux" *)
module _80_redstone_mux8 (A, B, S, Y);
    parameter WIDTH = 1;

    input [WIDTH-1:0] A, B;
    input S;
    output [WIDTH-1:0] Y;

    generate
        if (WIDTH == 8) begin
            RS_MUX8 _TECHMAP_REPLACE_ (.A(A), .B(B), .S(S), .Y(Y));
        end else if (WIDTH == 1) begin
            MUX _TECHMAP_REPLACE_ (.A(A), .B(B), .S(S), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$reduce_and" *)
module _80_redstone_reduce_and8 (A, Y);
    parameter A_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && Y_WIDTH == 1) begin
            RS_REDUCE_AND _TECHMAP_REPLACE_ (.A(A), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$reduce_or" *)
module _80_redstone_reduce_or8 (A, Y);
    parameter A_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && Y_WIDTH == 1) begin
            RS_REDUCE_OR _TECHMAP_REPLACE_ (.A(A), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

(* techmap_celltype = "$reduce_xor" *)
module _80_redstone_reduce_xor8 (A, Y);
    parameter A_SIGNED = 0;
    parameter A_WIDTH = 1;
    parameter Y_WIDTH = 1;

    input [A_WIDTH-1:0] A;
    output [Y_WIDTH-1:0] Y;

    generate
        if (A_WIDTH == 8 && Y_WIDTH == 1) begin
            RS_REDUCE_XOR _TECHMAP_REPLACE_ (.A(A), .Y(Y));
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule
