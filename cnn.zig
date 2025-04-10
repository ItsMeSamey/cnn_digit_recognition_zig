const std = @import("std");
const logger = @import("logger.zig");
const Layers = @import("layers.zig");

pub fn CNN(
  F: type,
  height: comptime_int, width: comptime_int,
  loss_gen: fn(LEN: comptime_int, F: type) type,
  layers: anytype,
) type {
  @setEvalBranchQuota(1000_000);
  const mergedTrainer = Layers.mergeAny(layers)(F, true, height, width);
  std.debug.assert(mergedTrainer.height == 1);

  const mergedTester = Layers.mergeAny(layers)(F, false, height, width);
  std.debug.assert(mergedTester.height == mergedTrainer.height);
  std.debug.assert(mergedTester.width == mergedTrainer.width);

  return struct {
    pub const OutputWidth = mergedTrainer.width;
    const LossFn = loss_gen(OutputWidth, F);

    pub const Gradient = mergedTrainer.layer.Gradient;

    fn getMaximalIndex(elements: anytype) usize {
      var max_index: usize = 0;
      inline for (0..elements.len) |i| {
        if (elements[i] > elements[max_index]) max_index = i;
      }
      return max_index;
    }

    fn copyData(dst_ptr: anytype, src_ptr: anytype) void {
      @setEvalBranchQuota(1000_000);
      const src_type = std.meta.Child(@TypeOf(src_ptr));
      const dst_type = std.meta.Child(@TypeOf(dst_ptr));

      if (src_type == dst_type) {
        dst_ptr.* = src_ptr.*;
        return;
      } else if (dst_type == void) {
        return;
      }

      const src_typeinfo = @typeInfo(src_type);
      const dst_typeinfo = @typeInfo(dst_type);

      if (comptime std.meta.activeTag(src_typeinfo) != std.meta.activeTag(dst_typeinfo)) {
        @compileError("Cant copy `" ++ @typeName(src_type) ++ "` to `" ++ @typeName(dst_type) ++ "` as they are different types entirely");
      }

      switch (dst_typeinfo) {
        .@"struct" => |info| {
          inline for (info.fields) |field| {
            copyData(&@field(dst_ptr, field.name), &@field(src_ptr, field.name));
          }
        },
        .@"array" => |info| {
          inline for (0..info.len) |i| {
            copyData(&@field(dst_ptr, i), &@field(src_ptr, i));
          }
        },
        else => {
          @compileError("Cant copy `" ++ @typeName(src_type) ++ "` to `" ++ @typeName(dst_type) ++ "` as they are incompatible");
        },
      }
    }

    pub const Trainer = struct {
      layer: mergedTrainer.layer,

      pub fn toTester(self: *const @This()) Tester {
        var retval: Tester = undefined;
        copyData(&retval, self);
        return retval;
      }

      // Sets the value in cache
      pub inline fn forward(self: *@This(), input: *[height][width]F, output: *[1][OutputWidth]F) void {
        @import("read_minst.zig").printImage(input);
        self.layer.forward(@ptrCast(input), @ptrCast(output));
      }

      pub fn backward(self: *@This(), input: *[height][width]F, output: *[1][OutputWidth]F, target: u8, gradient: *Gradient) void {
        var d_buf: [OutputWidth]F = undefined;
        LossFn.backward(&output[0], target, &d_buf);
        logger.log(&@src(), "predictions: {d}\n", .{&output[0]});
        logger.log(&@src(), "--- dLoss ({d}) ---\n\t{any}\n", .{target, &d_buf});

        self.layer.backward(@ptrCast(input), @ptrCast(output), undefined, @ptrCast(&d_buf), gradient, false);
      }

      pub fn reset(self: *@This(), rng: std.Random) void {
        self.layer.reset(rng);
      }

      pub const Options = struct {
        verbose: bool = true,
        batch_size: u32,
        learning_rate: F,
      };
      pub fn train(self: *@This(), iterator_readonly: anytype, options: Options) void {
        var iterator = iterator_readonly;
        var input_buf: [height][width]F = undefined;
        var output_buf: [1][OutputWidth]F = undefined;
        var gradients: Gradient = undefined;

        var step: usize = 0;
        while (true) {
          var n: usize = 0;
          var gross_loss: F = 0;

          step += 1;
          gradients.reset();

          for (0..options.batch_size) |_| {
            const next = iterator.next() orelse break;

            inline for (next.image, 0..) |row, i| {
              inline for (row, 0..) |val, j| {
                input_buf[i][j] = @as(F, @floatFromInt(val));
              }
            }

            n += 1;
            self.forward(&input_buf, &output_buf);
            self.backward(&input_buf, &output_buf, next.label, &gradients);

            const loss = LossFn.forward(&output_buf[0], next.label);
            gross_loss += loss;
            if (options.verbose) {
              logger.writer.print("{d:4}-{d:2} Loss({d} -> {d}) = {d:.3}\n", .{step, n, next.label, getMaximalIndex(output_buf[0]), loss*100, }) catch {};
              // logger.log(&@src(), "Gradients {any}\n", .{gradients});
            }
          }

          self.layer.applyGradient(&gradients, options.learning_rate / @as(F, @floatFromInt(options.batch_size)));

          logger.buffered.flush() catch {};
          if (!iterator.hasNext()) break;
        }
      }

      pub fn @"test"(self: *@This(), iterator_ro: anytype, comptime verbose: bool) F {
        defer logger.buffered.flush() catch {};
        var retval: usize = 0;
        var n: usize = 0;
        var iterator = iterator_ro;
        var input_buf: [height][width]F = undefined;
        var output_buf: [1][OutputWidth]F = undefined;

        while (iterator.next()) |next| {
          n += 1;
          inline for (next.image, 0..) |row, i| {
            inline for (row, 0..) |val, j| {
              input_buf[i][j] = @as(F, @floatFromInt(val));
            }
          }
          self.forward(&input_buf, &output_buf);

          const guess = getMaximalIndex(output_buf[0]);
          if (guess == next.label) retval += 1;

          if (verbose) {
            logger.writer.print("{d:5} Prediction({d}) = {d}\n", .{n, next.label, guess}) catch {};
          }
        }

        return @as(F, @floatFromInt(retval)) / @as(F, @floatFromInt(n));
      }
    };

    pub const Tester = struct {
      layer: mergedTester.layer,

      pub fn toTrainer(self: *const @This()) Trainer {
        var retval: Trainer = undefined;
        copyData(&retval, self);
        return retval;
      }

      pub fn forward(self: *const @This(), input: *[height][width]F, output: *[1][OutputWidth]F) void {
        self.layer.forward(@ptrCast(input), @ptrCast(output));
      }

      pub fn @"test"(self: *const @This(), iterator_ro: anytype, comptime verbose: bool) F {
        defer logger.buffered.flush() catch {};
        var retval: usize = 0;
        var n: usize = 0;
        var iterator = iterator_ro;
        var input_buf: [height][width]F = undefined;
        var output_buf: [1][OutputWidth]F = undefined;

        while (iterator.next()) |next| {
          n += 1;
          inline for (next.image, 0..) |row, i| {
            inline for (row, 0..) |val, j| {
              input_buf[i][j] = @as(F, @floatFromInt(val));
            }
          }
          self.forward(&input_buf, &output_buf);

          const guess = getMaximalIndex(output_buf[0]);
          if (guess == next.label) retval += 1;

          if (verbose) {
            logger.writer.print("{d:5} Prediction({d}) = {d}\n", .{n, next.label, guess}) catch {};
          }
        }

        return @as(F, @floatFromInt(retval)) / @as(F, @floatFromInt(n));
      }
    };
  };
}

fn testDeepnet(rinit: comptime_int) !void {
  const Layer = @import("layers.zig");
  const Activation = @import("functions_activate.zig");
  const Loss = @import("functions_loss.zig");

  const Iterator = struct {
    rng: std.Random,
    remaining: u32,
    val: [1][3]u8 = undefined,
    repetitions: u32,
    remaining_repetitions: u32 = 0,

    fn next(self: *@This()) ?struct {
      image: *[1][3]u8,
      label: u8,
    } {
      if (self.remaining_repetitions > 0) {
        self.remaining_repetitions -= 1;
      } else if (self.remaining > 0) {
        self.remaining -= 1;
        self.remaining_repetitions = self.repetitions;
        self.val[0][0] = self.rng.intRangeLessThan(u8, 0, 2);
        self.val[0][1] = self.rng.intRangeLessThan(u8, 0, 2);
        self.val[0][2] = self.rng.intRangeLessThan(u8, 0, 2);
      } else {
        return null;
      }

      return .{
        .image = &self.val,
        .label = (self.val[0][0] ^ self.val[0][1]) + self.val[0][2],
      };
    }

    pub fn hasNext(self: *const @This()) bool {
      return self.remaining_repetitions > 0 or self.remaining > 0;
    }
  };

  // xor test
  const cnn = CNN(f64, 1, 3, Loss.MeanSquaredError, [_]Layer.LayerType{
    Layer.getDense(6, Activation.getPReLU(0.1)),
    Layer.getDense(6, Activation.getPReLU(0.1)),
    Layer.getDense(3, Activation.NormalizeSquared),
  });

  var trainer: cnn.Trainer = undefined;

  var rng = std.Random.DefaultPrng.init(rinit);
  trainer.reset(rng.random());

  for (0..16) |i| {
    trainer.train(Iterator{.rng = rng.random(), .remaining = 1024, .repetitions = 0}, .{
      .verbose = false,
      .batch_size = @intCast(8),
      .learning_rate = @as(f64, 1.4) / @as(f64, @floatFromInt(1 + i)),
    });
  }

  var tester = trainer.toTester();
  const accuracy = tester.@"test"(Iterator{.rng = rng.random(), .remaining = 512, .repetitions = 0}, false);
  std.debug.print("\n>>{d} Final Accuracy: {d:.3}%\n", .{rinit, accuracy*100});
}

test "CNN Deepnet" {
  // inline for (0..64) |i| { try testDeepnet(i); }
  try testDeepnet(6);
}

