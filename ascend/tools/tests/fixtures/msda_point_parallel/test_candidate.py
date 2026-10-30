"""Source contracts and actual PyTorch CPU equivalence; no device operations."""
import ast
import hashlib
import importlib.util
import itertools
from pathlib import Path
import unittest

import baseline
import candidate

ROOT = Path(__file__).resolve().parent
BASE_SHA = 'feb35fc82b67dec1fbf8d24c0f5f60d6ff30356e8f419cf37e90d92481e854d9'
MEASUREMENTS = []
torch = None
if importlib.util.find_spec('torch') is not None:
    import torch


def nested(tree, name):
    matches = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(matches) != 1:
        raise AssertionError('Expected one exact source definition: ' + name)
    return matches[0]


def dump(node):
    return ast.dump(node, include_attributes=False)


def assignments(tree, names):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found.append((node.lineno, target.id, dump(node.value)))
    return [(name, value) for _, name, value in sorted(found)]


class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = ast.parse((ROOT / 'baseline.py').read_text(encoding='utf-8'))
        cls.b = ast.parse((ROOT / 'candidate.py').read_text(encoding='utf-8'))

    def test_original_whole_SHA_and_shared_coordinate_helpers(self):
        self.assertEqual(Path(baseline.__file__).resolve(), ROOT / 'baseline.py')
        self.assertEqual(Path(candidate.__file__).resolve(), ROOT / 'candidate.py')
        self.assertEqual(hashlib.sha256((ROOT / 'baseline.py').read_bytes()).hexdigest(), BASE_SHA)
        self.assertIs(candidate.canonical, baseline.canonical)
        self.assertIs(candidate.pixel_coordinates, baseline.pixel_coordinates)
        for name in ('canonical', 'integer_scale_parts', 'finite_padding_coordinate', 'compensated_scale_offset_FP32', 'pixel_coordinates'):
            self.assertTrue(nested(self.a, name))

    def test_all_scalar_coordinate_corner_and_bilinear_expression_ASTs(self):
        names = {'x0', 'y0', 'lx', 'ly', 'hx', 'hy', 'active', 'valid', 'address', 'w1', 'w2', 'w3', 'w4', 'val'}
        a = assignments(nested(self.a, 'samples'), names)
        b = assignments(nested(self.b, 'samples'), names)
        self.assertEqual(a, b)
        # An association/order change is visible to this gate.
        bad = ast.parse((ROOT / 'candidate.py').read_text(encoding='utf-8').replace('val = val + w3.unsqueeze(-1) * v3', 'val = val + w4.unsqueeze(-1) * v4'))
        self.assertNotEqual(a, assignments(nested(bad, 'samples'), names))

    def test_gradient_expression_AST_and_32_channel_sum_exact(self):
        names = {'tg', 'gh', 'gw', 'dx', 'dy'}
        self.assertEqual(assignments(nested(self.a, 'backward'), names), assignments(nested(self.b, 'backward'), names))
        self.assertEqual(dump(nested(self.a, 'serial_channels')), dump(nested(self.b, 'serial_channels')))
        loop = nested(self.b, 'serial_channels').body[1]
        self.assertEqual(ast.unparse(loop.iter), 'range(1, 32)')
        self.assertEqual(ast.unparse(loop.body[0]), 'output = output + tensor[..., channel]')

    def test_gather_has_only_added_point_axis(self):
        a = nested(self.a, 'samples')
        b = nested(self.b, 'samples')
        def gather_append(tree):
            return [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'pixels.append'][0]
        actual = ast.unparse(gather_append(b))
        self.assertEqual(actual.replace('h, locations.shape[4], c)', 'h, c)'), ast.unparse(gather_append(a)))
        for tree in (a, b):
            flat = [n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'flat'][0]
            self.assertEqual(ast.unparse(flat.value), 'value.reshape(b * s * h, c)')

    def test_forward_point_accumulation_not_reduced(self):
        forward = nested(self.b, 'forward')
        outer = [n for n in forward.body if isinstance(n, ast.For)][0]
        inner = outer.body[0]
        self.assertIsInstance(inner, ast.For)
        self.assertEqual(ast.unparse(inner.target), 'point')
        self.assertEqual(ast.unparse(inner.iter), 'range(locations_used.shape[4])')
        self.assertEqual(ast.unparse(inner.body[0]), 'output[:, begin:end].add_(val[..., point, :] * weight[..., point].unsqueeze(-1))')
        body = [n for n in forward.body if isinstance(n, ast.Assign)]
        canonical_line = next(n.lineno for n in body if isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'canonical')
        self.assertLess(canonical_line, outer.lineno)

    def test_value_gradient_point_then_corner_and_original_flatten(self):
        backward = nested(self.b, 'backward')
        flat_assignment = [n for n in ast.walk(backward) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'flat'][0]
        self.assertEqual(ast.unparse(flat_assignment.value), 'gv.view(b * s * h, c)')
        point = [n for n in ast.walk(backward) if isinstance(n, ast.For) and ast.unparse(n.target) == 'point'][0]
        corner = point.body[0]
        self.assertEqual(ast.unparse(corner.iter), 'zip(coef, addresses, masks)')
        self.assertEqual(ast.unparse(corner.body[0]), 'contribution = w[..., point].unsqueeze(-1) * tg[..., point, :]')
        self.assertEqual(ast.unparse(corner.body[1]), 'contribution = contribution * mask[..., point].unsqueeze(-1)')
        self.assertEqual(ast.unparse(corner.body[2]), 'flat.index_add_(0, addr[..., point].reshape(-1), contribution.reshape(-1, c))')
        chunk = [n for n in nested(self.b, 'samples').body if isinstance(n, ast.For)][0]
        self.assertEqual(ast.unparse(chunk.iter), 'range(0, q, chunk)')

    def test_legacy_tail_reconstruction_exact_and_CPU_scope(self):
        a = nested(self.a, 'backward')
        b = nested(self.b, 'backward')
        original = [n for n in a.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'ctx.legacy'][0]
        changed = [n for n in b.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'ctx.legacy'][0]
        self.assertEqual(dump(original), dump(changed))
        self.assertEqual(ast.unparse(nested(self.b, 'forward').body[0].value.args[0]), "value.device.type in ('cpu', 'npu')")
        calls = [ast.unparse(n.func) for n in ast.walk(self.b) if isinstance(n, ast.Call)]
        self.assertFalse(any(x.endswith(('.sum', '.matmul', '.einsum', '.compile', '.autocast')) for x in calls))


@unittest.skipUnless(torch is not None, 'Actual PyTorch CPU unavailable; source checks do not establish tensor equivalence')
class TensorEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise AssertionError('CPU-only suite requires CUDA not initialized')
        cls.a = baseline.make_function(torch)
        cls.b = candidate.make_function(torch)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU-only suite initialized CUDA')

    def inputs(self, b=2, q=11, h=2, pairs=((3, 4), (2, 2)), p=4, legacy=False, seed=731, noncontig=False):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        s = sum(y * x for y, x in pairs)
        l = 4 if legacy else len(pairs)
        value = torch.randn(b, s, h, 32, generator=gen, dtype=torch.float32)
        loc = torch.rand(b, q, h, l, p, 2, generator=gen, dtype=torch.float32) * 1.5 - 0.25
        weight = torch.randn(b, q, h, l, p, generator=gen, dtype=torch.float32)
        go = torch.randn(b, q, h * 32, generator=gen, dtype=torch.float32)
        if noncontig:
            def strided(x):
                z = torch.empty(tuple(x.shape[:-1]) + (x.shape[-1] * 2,), dtype=x.dtype)
                z[..., ::2].copy_(x)
                return z[..., ::2]
            value, loc, weight, go = map(strided, (value, loc, weight, go))
            self.assertTrue(all(not t.is_contiguous() for t in (value, loc, weight, go)))
        return value, loc, weight, go

    def run_one(self, fn, data, pairs, chunk, legacy, needed):
        inputs = [x.detach().requires_grad_(want) for x, want in zip(data[:3], needed)]
        output = fn.apply(*inputs, pairs, legacy, chunk)
        grads = [None, None, None]
        if any(needed):
            wanted = [x for x, want in zip(inputs, needed) if want]
            got = iter(torch.autograd.grad(output, wanted, data[3], create_graph=False))
            grads = [next(got) if want else None for want in needed]
        else:
            self.assertFalse(output.requires_grad)
        return (output, *grads)

    def compare(self, name, data, pairs=((3, 4), (2, 2)), chunk=4, legacy=False, needed=(True, True, True)):
        before = [x.detach().clone() for x in data]
        a = self.run_one(self.a, data, pairs, chunk, legacy, needed)
        b = self.run_one(self.b, data, pairs, chunk, legacy, needed)
        for label, left, right in zip(('forward', 'grad_value', 'grad_locations', 'grad_attention'), a, b):
            if left is None:
                self.assertIsNone(right)
                continue
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.dtype, torch.float32)
            self.assertEqual(right.dtype, torch.float32)
            self.assertEqual(left.device.type, 'cpu')
            self.assertEqual(right.device.type, 'cpu')
            self.assertTrue(torch.isfinite(left).all().item())
            self.assertTrue(torch.isfinite(right).all().item())
            lu = left.detach().contiguous().reshape(-1).view(torch.uint8)
            ru = right.detach().contiguous().reshape(-1).view(torch.uint8)
            raw_equal = torch.equal(lu, ru)
            absdiff = (left.detach() - right.detach()).abs()
            outside = absdiff > (1e-6 + 1e-5 * left.detach().abs())
            MEASUREMENTS.append({'case': name, 'component': label, 'shape': list(left.shape), 'elements': left.numel(), 'raw_equal': raw_equal, 'different_bytes': int((lu != ru).sum().item()), 'max_abs': float(absdiff.max().item()), 'outside_original_tolerance': int(outside.sum().item()), 'baseline_raw_sha256': hashlib.sha256(lu.numpy().tobytes()).hexdigest(), 'candidate_raw_sha256': hashlib.sha256(ru.numpy().tobytes()).hexdigest()})
            # Strict raw bytes are the CPU acceptance gate, not an allclose fallback.
            self.assertTrue(raw_equal, name + '/' + label + ': raw FP32 differs')
        for old, new in zip(before, data):
            self.assertTrue(torch.equal(old.reshape(-1).contiguous().view(torch.uint8), new.reshape(-1).contiguous().view(torch.uint8)), 'Input mutated')
        return a, b

    def test_multilevel_chunks_and_tails(self):
        data = self.inputs(q=13)
        for chunk in (1, 2, 5, 13, 2048):
            with self.subTest(chunk=chunk):
                self.compare('multilevel_chunk_' + str(chunk), data, chunk=chunk)

    def test_actual_2048_boundary_plus_tail(self):
        self.compare('q2051_chunk2048', self.inputs(b=1, q=2051), chunk=2048)

    def test_noncontiguous_all_inputs_and_GO(self):
        self.compare('noncontiguous', self.inputs(q=17, noncontig=True), chunk=7)

    def test_all_needs_input_grad_combinations(self):
        data = self.inputs(b=1, q=7, p=3)
        for needed in itertools.product((False, True), repeat=3):
            with self.subTest(needed=needed):
                self.compare('needs_' + ''.join(str(int(x)) for x in needed), data, chunk=3, needed=needed)

    def test_legacy_entire_prefix_before_chunk_and_zero_unused_tail(self):
        pairs = ((3, 4),)
        data = self.inputs(b=2, q=11, h=8, pairs=pairs, legacy=True)
        a, b = self.compare('legacy_B2Q11H8_rawL4', data, pairs, chunk=3, legacy=True)
        used = 2 * 11 * 8 * 4
        for result in (a, b):
            self.assertEqual(torch.count_nonzero(result[2].reshape(-1)[used * 2:]).item(), 0)
            self.assertEqual(torch.count_nonzero(result[3].reshape(-1)[used:]).item(), 0)
        altered = tuple(x.clone() for x in data)
        altered[1].reshape(-1)[used * 2:] = 3.125
        altered[2].reshape(-1)[used:] = -91.0
        aa, bb = self.compare('legacy_unused_tail_mutation', altered, pairs, chunk=3, legacy=True)
        for original, changed in zip(a, aa):
            self.assertTrue(torch.equal(original.contiguous().view(torch.uint8), changed.contiguous().view(torch.uint8)))
        # Full-prefix B/Q interpretation must not become raw[:, chunk, :, :1].
        interpreted, _ = baseline.canonical(data[0], data[1], data[2], pairs, True)
        self.assertFalse(torch.equal(interpreted, data[1][:, :, :, :1]))

    def test_borders_padding_signed_zero_and_nonfinite_locations(self):
        pairs = ((2, 3), (1, 1))
        data = self.inputs(b=1, q=15, h=1, pairs=pairs, p=4)
        coordinates = [-float('inf'), float('inf'), float('nan'), -9., -4., -1., -0.5, -0., 0., 1./6., 0.5, 5./6., 1., 1.5, 4.]
        for q, value in enumerate(coordinates):
            data[1][:, q, :, :, :, 0] = value
            data[1][:, q, :, :, :, 1] = coordinates[-1 - q]
        self.compare('border_nonfinite_guard', data, pairs, chunk=4)

    def test_supported_maximum_axis_single_point_and_integer_scale_domain(self):
        # In the supported 1..4096 integer domain the low component is zero:
        # <=4095 uses shift0; 4096 is exactly divisible by its split unit2.
        for axis in range(1, 4097):
            self.assertEqual(baseline.integer_scale_parts(axis), (axis, 0))
        with self.assertRaises(ValueError):
            baseline.integer_scale_parts(4097)
        pairs = ((1, 4096),)
        self.compare('axis4096_P1', self.inputs(b=1, q=7, h=1, pairs=pairs, p=1), pairs, chunk=3)

    def test_seven_points_three_levels(self):
        pairs = ((2, 3), (1, 2), (1, 1))
        self.compare('L3_P7', self.inputs(b=2, q=9, h=3, pairs=pairs, p=7), pairs, chunk=4)

    def test_colliding_addresses_cancellation_and_channel_order(self):
        pairs = ((2, 2), (1, 1))
        data = self.inputs(b=2, q=19, h=2, pairs=pairs, p=4)
        data[1].fill_(0.5)
        pattern = torch.tensor([16777216., 1., -16777216., 1.], dtype=torch.float32)
        data[2].copy_(pattern.expand_as(data[2]))
        data[3].reshape(-1).copy_(pattern.repeat(data[3].numel() // 4))
        data[0].reshape(-1).copy_(pattern.repeat(data[0].numel() // 4))
        self.compare('collision_cancellation', data, pairs, chunk=5)

    def test_same_invalid_geometry_dtype_and_legacy_rejected(self):
        original = self.inputs(b=1, q=3)
        cases = [([x.to(torch.float64) for x in original], False, 2), (original, True, 2), (original, False, 0), (original, False, 2049)]
        for data, legacy, chunk in cases:
            for fn in (self.a, self.b):
                with self.assertRaises(ValueError):
                    fn.apply(*data[:3], ((3, 4), (2, 2)), legacy, chunk)


if __name__ == '__main__':
    unittest.main(verbosity=2)
