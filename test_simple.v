// Simple test for placement and routing
// This design uses gates with different sizes:
// - NOT (2x2x2), AND (3x2x2), OR (3x2x2), XOR (4x3x2), DFF (3x3x2)
module test_simple(
    input wire clk,
    input wire a,
    input wire b,
    input wire c,
    output wire out1,
    output wire out2
);

// Various gates with different sizes
wire not_a;      // NOT gate: width=2, depth=2, height=2
wire and_ab;     // AND gate: width=3, depth=2, height=2
wire or_bc;      // OR gate: width=3, depth=2, height=2
wire xor_result; // XOR gate: width=4, depth=3, height=2
reg dff_out;     // DFF: width=3, depth=3, height=2

// Logic operations
assign not_a = ~a;                    // NOT gate
assign and_ab = a & b;                // AND gate
assign or_bc = b | c;                 // OR gate
assign xor_result = and_ab ^ or_bc;   // XOR gate

// Flip-flop
always @(posedge clk) begin
    dff_out <= xor_result;            // DFF
end

// Outputs
assign out1 = not_a & dff_out;        // Another AND gate
assign out2 = dff_out;

endmodule
