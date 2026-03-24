/**
 * Talisay AI — Scan History Page
 * Shows saved analysis results with filters, card grid, and detail modal.
 * Adapted from talisay_oil's history page, using talisay_ai's design system.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  Pressable,
  Image,
  StyleSheet,
  Platform,
  Modal,
  ActivityIndicator,
  Alert,
} from 'react-native';
import Animated, {
  FadeInUp,
  FadeInDown,
  FadeInLeft,
  ZoomIn,
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import { useResponsive } from '../hooks/useResponsive';
import { Spacing, Shadows, BorderRadius, Typography, Layout as LayoutConst } from '../constants/Layout';
import { historyService } from '../services/historyService';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// ─── Helpers ───
function formatDateLabel(isoDate) {
  const d = new Date(isoDate);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function getCategoryColor(cat) {
  switch (cat?.toUpperCase()) {
    case 'GREEN': return '#22c55e';
    case 'YELLOW': return '#eab308';
    case 'BROWN': return '#92400e';
    default: return '#6b7280';
  }
}

const OIL_ML_PER_FRUIT = {
  YELLOW: 0.05,
  BROWN: 0.03,
  GREEN: 0.01,
};

const OIL_SAFETY_FACTOR_BY_COLOR = {
  YELLOW: 0.75,
  BROWN: 0.7,
  GREEN: 0.6,
};

const formatMl = (value) => Number(value || 0).toFixed(2);
const formatCalc = (value, digits = 2) => Number(value || 0).toFixed(digits);

function getColorCountsFromEntry(entry) {
  if (!entry) return {};

  if (entry.multiFruit || entry.analysisType === 'multi_fruit') {
    if (entry.colorDistribution && Object.keys(entry.colorDistribution).length > 0) {
      return Object.entries(entry.colorDistribution).reduce((acc, [color, count]) => {
        acc[String(color || '').toUpperCase()] = Number(count) || 0;
        return acc;
      }, {});
    }

    if (Array.isArray(entry.fruits) && entry.fruits.length > 0) {
      return entry.fruits.reduce((acc, fruit) => {
        const color = String(fruit?.color || '').toUpperCase();
        if (!color) return acc;
        acc[color] = (acc[color] || 0) + 1;
        return acc;
      }, {});
    }
  }

  const singleColor = String(entry.category || '').toUpperCase();
  return singleColor ? { [singleColor]: 1 } : {};
}

function estimateMinimumOilMl(entry) {
  const colorCounts = getColorCountsFromEntry(entry);
  return Object.entries(colorCounts).reduce((sum, [color, count]) => {
    const perFruit = OIL_ML_PER_FRUIT[color] || 0;
    const safetyFactor = OIL_SAFETY_FACTOR_BY_COLOR[color] ?? 0.6;
    return sum + (perFruit * count * safetyFactor);
  }, 0);
}

function estimateConservativeOilMlByColor(color, count = 1) {
  const normalizedColor = String(color || '').toUpperCase();
  const perFruit = OIL_ML_PER_FRUIT[normalizedColor] || 0;
  const safetyFactor = OIL_SAFETY_FACTOR_BY_COLOR[normalizedColor] ?? 0.6;
  return perFruit * count * safetyFactor;
}

function OilEstimateDetails({ entry, colors, isDark, title = 'Oil Yield Prediction', compact = false }) {
  if (!entry) return null;

  const [showOilFormula, setShowOilFormula] = useState(false);
  const [showOilMlFormula, setShowOilMlFormula] = useState(false);
  const isMultiFruit = entry.multiFruit || entry.analysisType === 'multi_fruit';
  const averageOil = Number(entry.oilYieldPercent || entry.averageOilYield || 0);
  const minimumOilMl = estimateMinimumOilMl(entry);

  const dims = entry.dimensions || {};
  const fruitLength = dims.length_cm ?? dims.lengthCm;
  const fruitWidth = dims.width_cm ?? dims.widthCm;
  const kernelMass = dims.kernel_mass_g ?? dims.kernelWeightG;
  const hasDimensionData = fruitLength != null && fruitWidth != null;
  const sizeFactor = hasDimensionData ? ((fruitLength * fruitWidth) / (5 * 3.5)) : null;
  const simpleKernelEstimate = sizeFactor != null ? (0.4 * sizeFactor) : null;

  const oilCalculation = entry?.oil_calculation || entry?.raw?.oil_calculation || null;
  const calcInputs = oilCalculation?.inputs || {};
  const mlPredictions = oilCalculation?.model_predictions || null;
  const formulaComponents = oilCalculation?.formula_components || null;
  const multiFruitEntries = Array.isArray(entry?.fruits) ? entry.fruits : [];
  const multiOilValues = multiFruitEntries
    .map((f) => Number(f?.oil_yield_percent || f?.oilYieldPercent))
    .filter((v) => Number.isFinite(v));
  const colorCountsForMl = getColorCountsFromEntry(entry);
  const mlBreakdown = Object.entries(colorCountsForMl).map(([color, count]) => {
    const perFruit = OIL_ML_PER_FRUIT[color] || 0;
    const safety = OIL_SAFETY_FACTOR_BY_COLOR[color] ?? 0.6;
    const subtotal = perFruit * safety * count;
    return { color, count, perFruit, safety, subtotal };
  });

  return (
    <View style={[styles.dsDetailCard, { backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : '#fafafa', borderColor: colors.borderLight }]}> 
      <View style={styles.dsDetailCardHeader}>
        <View style={[styles.dsDetailIconWrap, { backgroundColor: '#22c55e15' }]}>
          <Ionicons name="water" size={13} color="#22c55e" />
        </View>
        <Text style={[styles.dsDetailCardTitle, { color: colors.text }]}>{title}</Text>
      </View>

      <View style={styles.dsDetailRow}>
        <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>{isMultiFruit ? 'Average' : 'Predicted'}</Text>
        <View style={styles.predictedValueWrap}>
          <Text style={[styles.oilYieldBig, { color: colors.text }]}>{formatCalc(averageOil, 1)}%</Text>
          <Pressable
            onPress={() => setShowOilFormula((prev) => !prev)}
            style={[styles.formulaToggleBtn, { borderColor: colors.borderLight, backgroundColor: colors.backgroundSecondary }]}
          >
            <Ionicons name={showOilFormula ? 'chevron-up' : 'chevron-down'} size={12} color={colors.textSecondary} />
            <Text style={[styles.formulaToggleText, { color: colors.textSecondary }]}>How this % is estimated</Text>
          </Pressable>
        </View>
      </View>

      {showOilFormula && (
        <View style={[styles.formulaBox, { backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : '#f8fafc', borderColor: colors.borderLight }]}> 
          <Text style={[styles.formulaTitle, { color: colors.text }]}>Actual percentage calculation</Text>
          {isMultiFruit ? (
            <>
              <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>For multiple fruits, each fruit gets its own oil prediction first, then the app shows the average.</Text>
              {multiOilValues.length > 0 ? (
                <>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Per-fruit oil yields: {multiOilValues.map((v) => `${formatCalc(v)}%`).join(', ')}</Text>
                  <Text style={[styles.formulaEquation, { color: colors.text }]}>Average Oil Yield % = ({multiOilValues.map((v) => formatCalc(v)).join(' + ')}) ÷ {multiOilValues.length} = {formatCalc(averageOil)}%</Text>
                </>
              ) : (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Average Oil Yield % = mean of all detected fruits' oil percentages.</Text>
              )}

              <Text style={[styles.formulaSubTitle, { color: colors.text }]}>Simple kernel mass estimate</Text>
              <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>For each fruit: Kernel mass = 0.4 × ((Length × Width) ÷ (5.0 × 3.5))</Text>
              {multiFruitEntries[0]?.dimensions?.length_cm != null && multiFruitEntries[0]?.dimensions?.width_cm != null && (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Example (Fruit #1): 0.4 × (({formatCalc(multiFruitEntries[0].dimensions.length_cm, 2)} × {formatCalc(multiFruitEntries[0].dimensions.width_cm, 2)}) ÷ 17.50) = {formatCalc(multiFruitEntries[0]?.dimensions?.kernel_mass_g, 3)} g</Text>
              )}
            </>
          ) : (
            <>
              {hasDimensionData ? (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Fruit size used: {formatCalc(fruitLength, 2)} cm × {formatCalc(fruitWidth, 2)} cm</Text>
              ) : (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Fruit size used: auto-estimated from image</Text>
              )}
              {kernelMass != null && (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Kernel amount used: {formatCalc(kernelMass, 3)} g</Text>
              )}

              {hasDimensionData && (
                <>
                  <Text style={[styles.formulaSubTitle, { color: colors.text }]}>Simple kernel mass estimate</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Size factor = (Length × Width) ÷ (5.0 × 3.5)</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>= ({formatCalc(fruitLength, 2)} × {formatCalc(fruitWidth, 2)}) ÷ 17.50 = {formatCalc(sizeFactor, 3)}</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Estimated kernel mass = 0.4 × Size factor</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>= 0.4 × {formatCalc(sizeFactor, 3)} = {formatCalc(simpleKernelEstimate, 3)} g</Text>
                </>
              )}

              {mlPredictions ? (
                <>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>1) Random Forest result = {formatCalc(mlPredictions.random_forest)}%</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>2) Gradient Boosting result = {formatCalc(mlPredictions.gradient_boosting)}%</Text>
                  <Text style={[styles.formulaEquation, { color: colors.text }]}>Final Oil Yield % = ({formatCalc(mlPredictions.random_forest)} + {formatCalc(mlPredictions.gradient_boosting)}) ÷ 2 = {formatCalc(averageOil)}%</Text>
                </>
              ) : formulaComponents ? (
                <>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Base from fruit color = {formatCalc(formulaComponents.base_from_color)}%</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Kernel size adjustment = {formatCalc(formulaComponents.kernel_adjustment)}%</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Length adjustment = {formatCalc(formulaComponents.length_adjustment)}%</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Shape adjustment = {formatCalc(formulaComponents.aspect_adjustment)}%</Text>
                  <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Weight adjustment = {formatCalc(formulaComponents.weight_adjustment)}%</Text>
                  <Text style={[styles.formulaEquation, { color: colors.text }]}>Final Oil Yield % = {formatCalc(formulaComponents.base_from_color)} + {formatCalc(formulaComponents.kernel_adjustment)} + {formatCalc(formulaComponents.length_adjustment)} + {formatCalc(formulaComponents.aspect_adjustment)} + {formatCalc(formulaComponents.weight_adjustment)} = {formatCalc(averageOil)}%</Text>
                </>
              ) : (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Calculation details are unavailable for this result source. The displayed value is the final percentage returned by the backend.</Text>
              )}

              {calcInputs?.color && (
                <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Color stage used: {String(calcInputs.color).toUpperCase()}</Text>
              )}
            </>
          )}
        </View>
      )}

      {minimumOilMl > 0 && (
        <View style={styles.dsDetailRow}>
          <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Estimated Minimum Oil</Text>
          <View style={styles.predictedValueWrap}>
            <Text style={[styles.dsDetailValue, { color: '#16a34a', textAlign: 'right' }]}>Around {formatMl(minimumOilMl)} ml of Oil can be extracted</Text>
            <Pressable
              onPress={() => setShowOilMlFormula((prev) => !prev)}
              style={[styles.formulaToggleBtn, { borderColor: colors.borderLight, backgroundColor: colors.backgroundSecondary }]}
            >
              <Ionicons name={showOilMlFormula ? 'chevron-up' : 'chevron-down'} size={12} color={colors.textSecondary} />
              <Text style={[styles.formulaToggleText, { color: colors.textSecondary }]}>How this ml is estimated</Text>
            </Pressable>
          </View>
        </View>
      )}

      {minimumOilMl > 0 && showOilMlFormula && (
        <View style={[styles.formulaBox, { backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : '#f8fafc', borderColor: colors.borderLight }]}> 
          <Text style={[styles.formulaTitle, { color: colors.text }]}>Actual ml calculation</Text>
          <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>The app estimates oil ml by fruit color, then applies a safety factor.</Text>
          {mlBreakdown.map((row, index) => (
            <Text key={`${row.color}-${index}`} style={[styles.formulaLine, { color: colors.textSecondary }]}>
              {row.color.charAt(0) + row.color.slice(1).toLowerCase()}: {row.count} × {formatMl(row.perFruit)} × {formatCalc(row.safety, 2)} = {formatMl(row.subtotal)} ml
            </Text>
          ))}
          {mlBreakdown.length > 0 ? (
            <Text style={[styles.formulaEquation, { color: colors.text }]}>Around ml = {mlBreakdown.map((row) => formatMl(row.subtotal)).join(' + ')} = {formatMl(minimumOilMl)} ml</Text>
          ) : (
            <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Around ml = no valid fruit colors detected.</Text>
          )}
          <Text style={[styles.formulaLine, { color: colors.textSecondary }]}>Base values used: Yellow 0.05 ml, Brown 0.03 ml, Green 0.01 ml per fruit.</Text>
        </View>
      )}

      {isMultiFruit && entry.oilYieldRange && (
        <View style={styles.dsDetailRow}>
          <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Range</Text>
          <Text style={[styles.dsDetailValue, { color: colors.text }]}>{entry.oilYieldRange[0]?.toFixed(1)}% – {entry.oilYieldRange[1]?.toFixed(1)}%</Text>
        </View>
      )}

      {!isMultiFruit && entry.yieldCategory && (
        <View style={styles.dsDetailRow}>
          <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Category</Text>
          <Text style={[styles.dsDetailValue, { color: colors.text }]}>{entry.yieldCategory}</Text>
        </View>
      )}

      <View style={styles.dsDetailRow}>
        <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Confidence</Text>
        <Text style={[styles.dsDetailValue, { color: colors.text }]}>
          {Math.round(((entry.oilConfidence || entry.overallConfidence || entry.confidence || 0) <= 1
            ? (entry.oilConfidence || entry.overallConfidence || entry.confidence || 0) * 100
            : (entry.oilConfidence || entry.overallConfidence || entry.confidence || 0)))}%
        </Text>
      </View>
    </View>
  );
}

// ─── History Card ───
function HistoryCard({ entry, onPress, delay = 0, colors, isDark, isDesktop }) {
  const scale = useSharedValue(1);
  const cardStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));
  const minimumOilMl = estimateMinimumOilMl(entry);
  const oilValue = entry.oilYieldPercent || entry.averageOilYield;

  return (
    <Animated.View entering={FadeInUp.delay(delay).duration(280)}>
      <AnimatedPressable
        onPress={onPress}
        onPressIn={() => { scale.value = withSpring(0.96); }}
        onPressOut={() => { scale.value = withSpring(1); }}
        style={[
          cardStyle,
          styles.historyCard,
          isDesktop ? styles.historyCardDesktop : styles.historyCardMobile,
          { backgroundColor: colors.card, borderColor: colors.borderLight },
        ]}
      >
        {/* Thumbnail */}
        {entry.imageUri ? (
          <Image source={{ uri: entry.imageUri }} style={styles.cardThumb} resizeMode="cover" />
        ) : (
          <View style={[styles.cardThumb, styles.cardThumbPlaceholder, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f1f5f9' }]}>
            <Ionicons name="leaf" size={28} color={colors.textTertiary} />
          </View>
        )}

        {/* Card Content */}
        <View style={styles.cardBody}>
          {/* Type badge */}
          {entry.analysisType === 'comparison' && (
            <View style={[styles.typeBadge, { backgroundColor: colors.primary + '12' }]}>
              <Ionicons name="git-compare" size={10} color={colors.primary} />
              <Text style={[styles.typeBadgeText, { color: colors.primary }]}>Comparison</Text>
            </View>
          )}
          {entry.analysisType === 'multi_fruit' && (
            <View style={[styles.typeBadge, { backgroundColor: '#7c3aed' + '12' }]}>
              <Ionicons name="apps" size={10} color="#7c3aed" />
              <Text style={[styles.typeBadgeText, { color: '#7c3aed' }]}>Multiple ({entry.fruitCount || '?'})</Text>
            </View>
          )}

          {/* Category */}
          <View style={[styles.catBadge, { backgroundColor: getCategoryColor(entry.category) }]}>
            <Text style={styles.catBadgeText}>{entry.category || 'Unknown'}</Text>
          </View>

          {/* Oil yield */}
          <Text style={[styles.oilYieldText, { color: colors.text }]}>
            {oilValue != null ? `${Number(oilValue).toFixed(1)}% oil` : '—'}
          </Text>
          {minimumOilMl > 0 && (
            <Text style={[styles.cardMinOilText, { color: '#16a34a' }]} numberOfLines={1}>
              Around {formatMl(minimumOilMl)} ml minimum
            </Text>
          )}

          {/* Comparison label */}
          {entry.comparisonLabel && (
            <Text style={[styles.compLabelText, { color: colors.textTertiary }]} numberOfLines={1}>
              {entry.comparisonLabel}
            </Text>
          )}

          {/* Date */}
          <Text style={[styles.cardDate, { color: colors.textSecondary }]}>{formatDateLabel(entry.createdAt)}</Text>
          <Text style={[styles.cardTime, { color: colors.textTertiary }]}>{formatTime(entry.createdAt)}</Text>
        </View>
      </AnimatedPressable>
    </Animated.View>
  );
}

// ─── Single Dataset Panel (matches Scan page ResultDisplay EXACTLY) ───
function DatasetPanel({ data, label, colors, isDark }) {
  if (!data) return null;
  const dims = data.dimensions || {};
  const minimumOilMl = estimateMinimumOilMl(data);
  const isBaseline = label === 'Baseline';
  const labelText = isBaseline ? 'EXISTING DATASET (BASELINE)' : 'YOUR OWN DATASET';
  const labelColor = isBaseline ? '#3b82f6' : colors.primary;
  const confRaw = data.colorConfidence;
  const confPercent = confRaw != null
    ? Math.round(typeof confRaw === 'number' && confRaw <= 1 ? confRaw * 100 : confRaw)
    : 0;

  return (
    <View style={[styles.dsPanel, { backgroundColor: colors.card, borderColor: colors.borderLight }]}>
      {/* Label badge */}
      <View style={[styles.dsLabelBadge, { backgroundColor: labelColor + '12' }]}>
        <Text style={[styles.dsLabelText, { color: labelColor }]}>{labelText}</Text>
      </View>

      {/* Image */}
      <View style={styles.dsImageCol}>
        {data.imageUri ? (
          <Image source={{ uri: data.imageUri }} style={styles.dsImage} resizeMode="contain" />
        ) : (
          <View style={[styles.dsImage, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f1f5f9', alignItems: 'center', justifyContent: 'center' }]}>
            <Ionicons name="leaf" size={32} color={colors.textTertiary} />
          </View>
        )}
        <Text style={[styles.dsFileName, { color: colors.textTertiary }]} numberOfLines={1}>
          {data.imageName || (isBaseline ? 'Existing Dataset' : 'Your Image')}
        </Text>
      </View>

      {/* Oil Yield — large text */}
      <View style={styles.dsYieldDisplay}>
        <Text style={[styles.dsYieldPercent, { color: colors.text }]}>
          {Math.round(data.oilYieldPercent || 0)}%
        </Text>
        <Text style={[styles.dsYieldLabel, { color: colors.textSecondary }]}>Oil Yield</Text>
        {minimumOilMl > 0 && (
          <Text style={[styles.dsMinOilText, { color: '#16a34a' }]}>Around {formatMl(minimumOilMl)} ml minimum</Text>
        )}
      </View>

      {/* Category Badge — full width bar */}
      <View style={[styles.dsCatBadge, { backgroundColor: getCategoryColor(data.category) }]}>
        <Text style={styles.dsCatBadgeText}>{data.category || 'Unknown'}</Text>
        <Text style={styles.dsCatBadgeConf}>{confPercent}%</Text>
      </View>

      <View style={styles.dsDetailsWrap}>
          {/* Color Classification */}
          <View style={[styles.dsDetailCard, { backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : '#fafafa', borderColor: colors.borderLight }]}>
            <View style={styles.dsDetailCardHeader}>
              <View style={[styles.dsDetailIconWrap, { backgroundColor: colors.primary + '15' }]}>
                <Ionicons name="color-palette" size={13} color={colors.primary} />
              </View>
              <Text style={[styles.dsDetailCardTitle, { color: colors.text }]}>Color Classification</Text>
            </View>
            <View style={styles.dsDetailRow}>
              <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Detected</Text>
              <Text style={[styles.dsDetailValue, { color: getCategoryColor(data.category) }]}>{data.category}</Text>
            </View>
            <View style={styles.dsDetailRow}>
              <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Confidence</Text>
              <Text style={[styles.dsDetailValue, { color: colors.text }]}>{confPercent}%</Text>
            </View>
            {data.maturityStage && (
              <View style={styles.dsDetailRow}>
                <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Maturity</Text>
                <Text style={[styles.dsDetailValue, { color: colors.text }]}>{data.maturityStage}</Text>
              </View>
            )}
          </View>

          {/* Dimensions */}
          {dims && Object.keys(dims).length > 0 && (
            <View style={[styles.dsDetailCard, { backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : '#fafafa', borderColor: colors.borderLight }]}>
              <View style={styles.dsDetailCardHeader}>
                <View style={[styles.dsDetailIconWrap, { backgroundColor: '#3b82f615' }]}>
                  <Ionicons name="resize" size={13} color="#3b82f6" />
                </View>
                <Text style={[styles.dsDetailCardTitle, { color: colors.text }]}>Dimensions</Text>
              </View>
              <View style={styles.dsDimGrid}>
                {[
                  { label: 'Length', value: dims.lengthCm ?? dims.length_cm, unit: 'cm' },
                  { label: 'Width', value: dims.widthCm ?? dims.width_cm, unit: 'cm' },
                  { label: 'Kernel', value: dims.kernelWeightG ?? dims.kernel_mass_g, unit: 'g', dec: 2 },
                  { label: 'Fruit', value: dims.wholeFruitWeightG ?? dims.whole_fruit_weight_g, unit: 'g', dec: 1 },
                ].map((d, i) => (
                  <View key={i} style={[styles.dsDimCell, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#fff' }]}>
                    <Text style={[styles.dsDimCellLabel, { color: colors.textTertiary }]}>{d.label}</Text>
                    <Text style={[styles.dsDimCellValue, { color: colors.text }]}>
                      {d.value != null ? Number(d.value).toFixed(d.dec ?? 1) : '—'} {d.unit}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          )}

          <OilEstimateDetails entry={data} colors={colors} isDark={isDark} title="Oil Yield Prediction" compact />

          {/* Total Images (for baseline) */}
          {data.totalImages > 0 && (
            <View style={[styles.dsDetailCard, { backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : '#fafafa', borderColor: colors.borderLight }]}>
              <View style={styles.dsDetailCardHeader}>
                <View style={[styles.dsDetailIconWrap, { backgroundColor: '#3b82f615' }]}>
                  <Ionicons name="server" size={13} color="#3b82f6" />
                </View>
                <Text style={[styles.dsDetailCardTitle, { color: colors.text }]}>Dataset Info</Text>
              </View>
              <View style={styles.dsDetailRow}>
                <Text style={[styles.dsDetailLabel, { color: colors.textSecondary }]}>Images Analyzed</Text>
                <Text style={[styles.dsDetailValue, { color: colors.text }]}>{data.totalImages}</Text>
              </View>
            </View>
          )}

          {/* Interpretation */}
          {data.interpretation && (
            <View style={[styles.dsDetailCard, { backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : '#fafafa', borderColor: colors.borderLight }]}>
              <View style={styles.dsDetailCardHeader}>
                <View style={[styles.dsDetailIconWrap, { backgroundColor: '#f59e0b15' }]}>
                  <Ionicons name="bulb" size={13} color="#f59e0b" />
                </View>
                <Text style={[styles.dsDetailCardTitle, { color: colors.text }]}>Interpretation</Text>
              </View>
              <Text style={[styles.dsInterpText, { color: colors.textSecondary }]}>{data.interpretation}</Text>
            </View>
          )}
        </View>
    </View>
  );
}

// ─── Multi-Fruit Result View (exact mirror of Scan page ResultDisplay for multiFruit) ───
function MultiFruitResultView({ entry, onDelete, colors, isDark }) {
  const oilYield = (entry.averageOilYield || entry.oilYieldPercent || 0).toFixed(1);
  const minimumOilMl = estimateMinimumOilMl(entry);
  const confidence = Math.round(
    (entry.overallConfidence || entry.colorConfidence || entry.confidence || 0) * 100
  );

  return (
    <>
      {/* Image */}
      <View style={styles.modalImageSection}>
        {entry.imageUri ? (
          <Image source={{ uri: entry.imageUri }} style={styles.modalImage} resizeMode="contain" />
        ) : (
          <View style={[styles.modalImagePlaceholder, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f1f5f9' }]}>
            <Ionicons name="leaf" size={48} color={colors.textTertiary} />
          </View>
        )}
      </View>

      <View style={styles.modalDetails}>
        {/* ── Stats card (mirrors scan StatsContent for multiFruit) ── */}
        <View style={[{ borderRadius: 14, borderWidth: 1, padding: 16, marginBottom: 12, backgroundColor: colors.card, borderColor: colors.borderLight }]}>
          {/* Big oil yield number */}
          <View style={{ alignItems: 'center', paddingBottom: 14 }}>
            <Text style={{ fontSize: 52, fontWeight: '800', color: colors.text, letterSpacing: -2 }}>{oilYield}%</Text>
            <Text style={{ fontSize: 13, color: colors.textSecondary, marginTop: 2 }}>Avg. Oil Yield</Text>
            {minimumOilMl > 0 && (
              <Text style={{ fontSize: 12, color: '#16a34a', marginTop: 4, fontWeight: '700' }}>
                Around {formatMl(minimumOilMl)} ml of Oil can be extracted
              </Text>
            )}
          </View>
          {/* Fruit count badge */}
          <View style={{ flexDirection: 'row', justifyContent: 'center', marginBottom: 14 }}>
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: '#7c3aed', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 5 }}>
              <Ionicons name="apps" size={14} color="#fff" />
              <Text style={{ color: '#fff', fontWeight: '700', fontSize: 13 }}>{entry.fruitCount} Fruits</Text>
            </View>
          </View>
          {/* Color distribution */}
          {entry.colorDistribution && Object.keys(entry.colorDistribution).length > 0 && (
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 10, marginBottom: 14 }}>
              {Object.entries(entry.colorDistribution).map(([c, n]) => (
                <View key={c} style={{ flexDirection: 'row', alignItems: 'center', gap: 5 }}>
                  <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: getCategoryColor(c.toUpperCase()) }} />
                  <Text style={{ fontSize: 13, color: colors.text, fontWeight: '600' }}>
                    {c.charAt(0).toUpperCase() + c.slice(1)}: {n}
                  </Text>
                </View>
              ))}
            </View>
          )}
          {/* Range row */}
          {entry.oilYieldRange && (
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderTopWidth: 1, borderTopColor: colors.borderLight }}>
              <Text style={{ fontSize: 13, color: colors.textSecondary }}>Range</Text>
              <Text style={{ fontSize: 13, color: colors.text, fontWeight: '600' }}>
                {entry.oilYieldRange[0]?.toFixed(1)}–{entry.oilYieldRange[1]?.toFixed(1)}%
              </Text>
            </View>
          )}
          {/* Confidence row */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderTopWidth: 1, borderTopColor: colors.borderLight }}>
            <Text style={{ fontSize: 13, color: colors.textSecondary }}>Confidence</Text>
            <Text style={{ fontSize: 13, color: colors.text, fontWeight: '600' }}>{confidence}%</Text>
          </View>
        </View>

        {/* Per-fruit breakdown */}
            {entry.fruits?.length > 0 && (
              <View style={[styles.modalSection, { borderColor: colors.borderLight }]}>
                <View style={styles.modalSectionHeader}>
                  <Ionicons name="apps" size={16} color="#7c3aed" />
                  <Text style={[styles.modalSectionTitle, { color: colors.text }]}>{entry.fruitCount} Fruits — Details</Text>
                </View>
                {entry.fruits.map((f, i) => (
                  <View key={i} style={[{ borderRadius: 8, padding: 10, marginBottom: 6, backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)' }]}>
                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: getCategoryColor(f.color?.toUpperCase()), borderRadius: 12, paddingHorizontal: 10, paddingVertical: 3 }}>
                        <Text style={{ color: '#fff', fontWeight: '700', fontSize: 11 }}>#{f.fruit_index ?? (i + 1)} {f.color?.toUpperCase()}</Text>
                      </View>
                      <Text style={{ fontSize: 14, fontWeight: '700', color: '#22c55e' }}>
                        {f.oil_yield_percent?.toFixed(1)}% oil · Around {formatMl(estimateConservativeOilMlByColor(f.color, 1))} ml
                      </Text>
                    </View>
                    {f.dimensions && (
                      <Text style={{ fontSize: 11, color: colors.textTertiary, marginTop: 2 }}>
                        L {f.dimensions.length_cm?.toFixed(1)} cm · W {f.dimensions.width_cm?.toFixed(1)} cm · {f.dimensions_source || 'estimated'}
                      </Text>
                    )}
                    {f.confidence != null && (
                      <Text style={{ fontSize: 11, color: colors.textTertiary }}>
                        Conf: {Math.round((f.confidence || 0) * 100)}%
                      </Text>
                    )}
                  </View>
                ))}
              </View>
            )}

            <OilEstimateDetails entry={entry} colors={colors} isDark={isDark} title="Average Oil Yield" />

            {/* Interpretation */}
            {entry.interpretation && (
              <View style={[styles.modalSection, { borderColor: colors.borderLight }]}>
                <View style={styles.modalSectionHeader}>
                  <Ionicons name="bulb" size={16} color="#f59e0b" />
                  <Text style={[styles.modalSectionTitle, { color: colors.text }]}>Interpretation</Text>
                </View>
                <Text style={[styles.interpText, { color: colors.textSecondary }]}>{entry.interpretation}</Text>
              </View>
            )}

        {/* Delete */}
        <Pressable onPress={onDelete} style={[styles.deleteBtn, { borderColor: '#ef444440' }]}>
          <Ionicons name="trash-outline" size={16} color="#ef4444" />
          <Text style={styles.deleteBtnText}>Delete This Entry</Text>
        </Pressable>
      </View>
    </>
  );
}

// ─── Detail Modal ───
function DetailModal({ entry, visible, onClose, onDelete, colors, isDark, allEntries }) {
  const { isDesktop } = useResponsive();
  if (!entry) return null;

  // Check if this is a comparison entry
  const isComparison = entry.analysisType === 'comparison';
  const isMultiFruit = entry.analysisType === 'multi_fruit' || entry.multiFruit;

  // Try to build baseline and own dataset data for comparison view
  let baselineData = null;
  let ownData = null;

  if (isComparison) {
    if (entry.comparisonLabel === 'Own Dataset' && entry.baselineData) {
      // This entry IS the own dataset and carries embedded baseline
      ownData = entry;
      baselineData = entry.baselineData;
    } else if (entry.comparisonId && allEntries) {
      // Find the paired entry by comparisonId
      const paired = allEntries.find(
        (e) => e.comparisonId === entry.comparisonId && (e.id || e._id) !== (entry.id || entry._id)
      );
      if (entry.comparisonLabel?.startsWith('Baseline')) {
        baselineData = entry;
        ownData = paired || null;
      } else {
        ownData = entry;
        baselineData = paired || entry.baselineData || null;
      }
    } else if (entry.comparisonLabel?.startsWith('Baseline')) {
      baselineData = entry;
    } else {
      ownData = entry;
      baselineData = entry.baselineData || null;
    }
  }

  const showComparison = isComparison && (baselineData || ownData);
  const dims = entry.dimensions || {};

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.modalBg}>
        <Pressable style={styles.modalBackdrop} onPress={onClose} />
        <View style={[styles.modalContainer, { backgroundColor: colors.card }]}>
          {/* Close button */}
          <Pressable onPress={onClose} style={styles.modalCloseBtn}>
            <Ionicons name="close" size={22} color="#fff" />
          </Pressable>

          <ScrollView
            style={styles.modalScroll}
            contentContainerStyle={styles.modalScrollContent}
            showsVerticalScrollIndicator={false}
            nestedScrollEnabled
            keyboardShouldPersistTaps="handled"
          >
            {/* Header */}
            <View style={[styles.modalHeader, { backgroundColor: isDark ? 'rgba(255,255,255,0.03)' : '#f8fafc', borderBottomColor: colors.borderLight }]}>
              <Text style={[styles.modalTitle, { color: colors.text }]} numberOfLines={1}>
                {showComparison ? 'Side-by-Side Comparison' : (entry.imageName || 'Analysis Result')}
              </Text>
              <View style={styles.modalMetaRow}>
                {isComparison && (
                  <View style={[styles.modalTypeBadge, { backgroundColor: colors.primary + '12' }]}>
                    <Ionicons name="git-compare" size={11} color={colors.primary} />
                    <Text style={[styles.modalTypeText, { color: colors.primary }]}>Comparison</Text>
                  </View>
                )}
                {isMultiFruit && (
                  <View style={[styles.modalTypeBadge, { backgroundColor: '#7c3aed' + '12' }]}>
                    <Ionicons name="apps" size={11} color="#7c3aed" />
                    <Text style={[styles.modalTypeText, { color: '#7c3aed' }]}>Multiple ({entry.fruitCount || '?'})</Text>
                  </View>
                )}
                <Text style={[styles.modalDate, { color: colors.textTertiary }]}>
                  {formatDateLabel(entry.createdAt)} • {formatTime(entry.createdAt)}
                </Text>
              </View>
            </View>

            {showComparison ? (
              /* ─── Comparison View (Side-by-Side) ─── */
              <View style={styles.modalDetails}>
                <View style={[styles.comparisonRow, !isDesktop && styles.comparisonCol]}>
                  {baselineData && (
                    <DatasetPanel data={baselineData} label="Baseline" colors={colors} isDark={isDark} />
                  )}
                  {ownData && (
                    <DatasetPanel data={ownData} label="Own Dataset" colors={colors} isDark={isDark} />
                  )}
                </View>

                {/* Comparison Summary */}
                {baselineData && ownData && baselineData.oilYieldPercent != null && ownData.oilYieldPercent != null && (
                  <View style={[styles.compSummaryBox, { backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : '#f0f9ff', borderColor: colors.borderLight }]}>
                    <View style={styles.modalSectionHeader}>
                      <Ionicons name="analytics" size={16} color="#3b82f6" />
                      <Text style={[styles.modalSectionTitle, { color: colors.text }]}>Comparison Summary</Text>
                    </View>
                    {(() => {
                      const diff = ownData.oilYieldPercent - baselineData.oilYieldPercent;
                      const absDiff = Math.abs(diff).toFixed(1);
                      const icon = diff > 0 ? 'arrow-up' : diff < 0 ? 'arrow-down' : 'remove';
                      const diffColor = diff > 0 ? '#22c55e' : diff < 0 ? '#ef4444' : '#6b7280';
                      const catMatch = (baselineData.category || '').toUpperCase() === (ownData.category || '').toUpperCase();
                      return (
                        <View style={{ gap: 8, marginTop: 8 }}>
                          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                            <Ionicons name={icon} size={16} color={diffColor} />
                            <Text style={{ fontSize: 14, fontWeight: '700', color: diffColor }}>
                              {absDiff}% {diff > 0 ? 'higher' : diff < 0 ? 'lower' : 'same'} oil yield
                            </Text>
                          </View>
                          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                            <Ionicons name={catMatch ? 'checkmark-circle' : 'close-circle'} size={16} color={catMatch ? '#22c55e' : '#f59e0b'} />
                            <Text style={{ fontSize: 13, color: colors.textSecondary }}>
                              Category: {catMatch ? 'Same' : 'Different'} ({baselineData.category} vs {ownData.category})
                            </Text>
                          </View>
                          <View style={{ flexDirection: 'row', justifyContent: 'space-between', borderTopWidth: 1, borderTopColor: colors.borderLight, paddingTop: 6 }}>
                            <Text style={{ fontSize: 13, color: '#2563eb', fontWeight: '600' }}>Baseline Minimum</Text>
                            <Text style={{ fontSize: 13, color: '#2563eb', fontWeight: '700' }}>Around {formatMl(estimateMinimumOilMl(baselineData))} ml</Text>
                          </View>
                          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                            <Text style={{ fontSize: 13, color: '#16a34a', fontWeight: '600' }}>Your Image Minimum</Text>
                            <Text style={{ fontSize: 13, color: '#16a34a', fontWeight: '700' }}>Around {formatMl(estimateMinimumOilMl(ownData))} ml</Text>
                          </View>
                        </View>
                      );
                    })()}
                  </View>
                )}

                {/* Delete button */}
                <Pressable onPress={onDelete} style={[styles.deleteBtn, { borderColor: '#ef444440' }]}>
                  <Ionicons name="trash-outline" size={16} color="#ef4444" />
                  <Text style={styles.deleteBtnText}>Delete This Entry</Text>
                </Pressable>
              </View>
            ) : isMultiFruit ? (
              <MultiFruitResultView entry={entry} onDelete={onDelete} colors={colors} isDark={isDark} />
            ) : (
              /* ─── Single Analysis View ─── */
              <>
                {/* Image */}
                <View style={styles.modalImageSection}>
                  {entry.imageUri ? (
                    <Image source={{ uri: entry.imageUri }} style={styles.modalImage} resizeMode="contain" />
                  ) : (
                    <View style={[styles.modalImagePlaceholder, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f1f5f9' }]}>
                      <Ionicons name="leaf" size={48} color={colors.textTertiary} />
                    </View>
                  )}
                  {entry.comparisonLabel && (
                    <View style={[styles.compLabelBox, { backgroundColor: 'rgba(0,0,0,0.6)' }]}>
                      <Text style={styles.compLabelBoxText}>{entry.comparisonLabel}</Text>
                    </View>
                  )}
                </View>

                {/* Results */}
                <View style={styles.modalDetails}>
                  {/* Oil Yield + Color Row */}
                  <View style={styles.modalResultRow}>
                    <View style={[styles.modalResultCol, { backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : '#f0fdf4', borderColor: colors.borderLight }]}>
                      <Ionicons name="water" size={18} color="#22c55e" />
                      <Text style={[styles.modalResultLabel, { color: colors.textSecondary }]}>{isMultiFruit ? 'Avg. Oil Yield' : 'Oil Yield'}</Text>
                      <Text style={[styles.modalResultValue, { color: colors.text }]}>
                        {(entry.oilYieldPercent || entry.averageOilYield)?.toFixed(1) ?? '—'}%
                      </Text>
                      {estimateMinimumOilMl(entry) > 0 && (
                        <Text style={[styles.modalResultSub, { color: '#16a34a', fontWeight: '700' }]}>
                          Around {formatMl(estimateMinimumOilMl(entry))} ml minimum
                        </Text>
                      )}
                      {entry.yieldCategory && (
                        <Text style={[styles.modalResultSub, { color: colors.textTertiary }]}>{entry.yieldCategory}</Text>
                      )}
                      {isMultiFruit && entry.oilYieldRange && (
                        <Text style={[styles.modalResultSub, { color: colors.textTertiary }]}>
                          Range: {entry.oilYieldRange[0]?.toFixed(1)}–{entry.oilYieldRange[1]?.toFixed(1)}%
                        </Text>
                      )}
                    </View>
                    <View style={[styles.modalResultCol, { backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : '#fefce8', borderColor: colors.borderLight }]}>
                      {isMultiFruit ? (
                        <>
                          <Ionicons name="apps" size={18} color="#7c3aed" />
                          <Text style={[styles.modalResultLabel, { color: colors.textSecondary }]}>Fruits</Text>
                          <Text style={[styles.modalResultValue, { color: colors.text }]}>{entry.fruitCount || '?'}</Text>
                          {entry.colorDistribution && (
                            <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 4, marginTop: 4 }}>
                              {Object.entries(entry.colorDistribution).map(([c, n]) => (
                                <View key={c} style={{ flexDirection: 'row', alignItems: 'center', gap: 3 }}>
                                  <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: getCategoryColor(c.toUpperCase()) }} />
                                  <Text style={{ fontSize: 10, color: colors.textSecondary, fontWeight: '600' }}>{c}: {n}</Text>
                                </View>
                              ))}
                            </View>
                          )}
                        </>
                      ) : (
                        <>
                          <Ionicons name="color-palette" size={18} color={getCategoryColor(entry.category)} />
                          <Text style={[styles.modalResultLabel, { color: colors.textSecondary }]}>Color</Text>
                          <View style={[styles.modalCatBadge, { backgroundColor: getCategoryColor(entry.category) }]}>
                            <Text style={styles.modalCatText}>{entry.category || 'Unknown'}</Text>
                          </View>
                          {entry.colorConfidence != null && (
                            <Text style={[styles.modalResultSub, { color: colors.textTertiary }]}>
                              {Math.round(entry.colorConfidence * 100)}% confidence
                            </Text>
                          )}
                        </>
                      )}
                    </View>
                  </View>

                  {/* Multi-fruit per-fruit breakdown */}
                  {isMultiFruit && entry.fruits?.length > 0 && (
                    <View style={[styles.modalSection, { borderColor: colors.borderLight }]}>
                      <View style={styles.modalSectionHeader}>
                        <Ionicons name="list" size={16} color="#7c3aed" />
                        <Text style={[styles.modalSectionTitle, { color: colors.text }]}>Per-Fruit Details</Text>
                      </View>
                      {entry.fruits.map((f, i) => (
                        <View key={i} style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 6, borderBottomWidth: i < entry.fruits.length - 1 ? 1 : 0, borderBottomColor: colors.borderLight }}>
                          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                            <View style={[styles.modalCatBadge, { backgroundColor: getCategoryColor(f.color?.toUpperCase()), paddingHorizontal: 8, paddingVertical: 2 }]}>
                              <Text style={{ color: '#fff', fontSize: 10, fontWeight: '700' }}>#{f.fruit_index ?? (i + 1)} {f.color?.toUpperCase()}</Text>
                            </View>
                          </View>
                          <Text style={{ fontSize: 14, fontWeight: '700', color: '#22c55e' }}>
                            {f.oil_yield_percent?.toFixed(1)}% oil · Around {formatMl(estimateConservativeOilMlByColor(f.color, 1))} ml
                          </Text>
                        </View>
                      ))}
                    </View>
                  )}

                  <OilEstimateDetails entry={entry} colors={colors} isDark={isDark} title={isMultiFruit ? 'Average Oil Yield' : 'Oil Yield Prediction'} />

                  {/* Dimensions */}
                  {dims && Object.keys(dims).length > 0 && (
                    <View style={[styles.modalSection, { borderColor: colors.borderLight }]}>
                      <View style={styles.modalSectionHeader}>
                        <Ionicons name="resize" size={16} color={colors.primary} />
                        <Text style={[styles.modalSectionTitle, { color: colors.text }]}>Dimensions</Text>
                      </View>
                      <View style={styles.dimGrid}>
                        {[
                          { label: 'Length', value: dims.lengthCm ?? dims.length_cm, unit: 'cm', icon: 'resize-outline' },
                          { label: 'Width', value: dims.widthCm ?? dims.width_cm, unit: 'cm', icon: 'resize-outline' },
                          { label: 'Kernel', value: dims.kernelWeightG ?? dims.kernel_mass_g, unit: 'g', icon: 'scale-outline', decimals: 2 },
                          { label: 'Total', value: dims.wholeFruitWeightG ?? dims.whole_fruit_weight_g, unit: 'g', icon: 'barbell-outline', decimals: 1 },
                        ].map((d, i) => (
                          <View key={i} style={[styles.dimItem, { backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : '#f8fafc', borderColor: colors.borderLight }]}>
                            <Ionicons name={d.icon} size={16} color={colors.primary} />
                            <Text style={[styles.dimLabel, { color: colors.textTertiary }]}>{d.label}</Text>
                            <Text style={[styles.dimValue, { color: colors.text }]}>
                              {d.value != null ? Number(d.value).toFixed(d.decimals ?? 1) : '—'}
                            </Text>
                            <Text style={[styles.dimUnit, { color: colors.textTertiary }]}>{d.unit}</Text>
                          </View>
                        ))}
                      </View>
                    </View>
                  )}

                  {/* Maturity & Features */}
                  {(entry.maturityStage || entry.hasSpots) && (
                    <View style={[styles.modalSection, { borderColor: colors.borderLight }]}>
                      <View style={styles.modalSectionHeader}>
                        <Ionicons name="information-circle" size={16} color={colors.primary} />
                        <Text style={[styles.modalSectionTitle, { color: colors.text }]}>Features</Text>
                      </View>
                      {entry.maturityStage && (
                        <View style={styles.featureRow}>
                          <Text style={[styles.featureLabel, { color: colors.textSecondary }]}>Maturity:</Text>
                          <Text style={[styles.featureValue, { color: colors.text }]}>{entry.maturityStage}</Text>
                        </View>
                      )}
                      {entry.hasSpots && (
                        <View style={[styles.spotBadge, { backgroundColor: '#f9731612' }]}>
                          <Ionicons name="warning" size={14} color="#f97316" />
                          <Text style={{ color: '#f97316', fontSize: 12, fontWeight: '600' }}>
                            Spots detected ({entry.spotCoverage ? (entry.spotCoverage * 100).toFixed(1) : '—'}% coverage)
                          </Text>
                        </View>
                      )}
                    </View>
                  )}

                  {/* Interpretation */}
                  {entry.interpretation && (
                    <View style={[styles.modalSection, { borderColor: colors.borderLight }]}>
                      <View style={styles.modalSectionHeader}>
                        <Ionicons name="bulb" size={16} color="#f59e0b" />
                        <Text style={[styles.modalSectionTitle, { color: colors.text }]}>Interpretation</Text>
                      </View>
                      <Text style={[styles.interpText, { color: colors.textSecondary }]}>{entry.interpretation}</Text>
                    </View>
                  )}

                  {/* Delete button */}
                  <Pressable onPress={onDelete} style={[styles.deleteBtn, { borderColor: '#ef444440' }]}>
                    <Ionicons name="trash-outline" size={16} color="#ef4444" />
                    <Text style={styles.deleteBtnText}>Delete This Entry</Text>
                  </Pressable>
                </View>
              </>
            )}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

// ════════════════════════════════════════════════
// ─── MAIN HISTORY PAGE ───
// ════════════════════════════════════════════════
export default function HistoryPage() {
  const router = useRouter();
  const { colors, isDark } = useTheme();
  const { isAuthenticated } = useAuth();
  const { isMobile, isDesktop } = useResponsive();

  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalEntry, setModalEntry] = useState(null);
  const [filterType, setFilterType] = useState('all');

  // ─── Load history ───
  const loadHistory = useCallback(async () => {
    if (!isAuthenticated) {
      setEntries([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const { items } = await historyService.listHistory({ limit: 100 });
      const processed = (items || []).map((item) => ({
        ...item,
        analysisType: item.analysisType || 'single',
      }));
      setEntries(processed);
    } catch (e) {
      console.warn('[History]', e?.message);
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // ─── Actions ───
  const handleDelete = async (id) => {
    const doDelete = async () => {
      try {
        await historyService.deleteHistoryItem(id);
        setModalEntry(null);
        loadHistory();
      } catch (e) {
        console.warn('[History] delete error', e);
      }
    };

    if (Platform.OS === 'web') {
      if (window.confirm('Delete this scan entry?')) doDelete();
    } else {
      Alert.alert('Delete Entry', 'Are you sure?', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: doDelete },
      ]);
    }
  };

  const handleClear = async () => {
    const doClear = async () => {
      try {
        await historyService.clearAllHistory();
        setEntries([]);
      } catch (e) {
        console.warn('[History] clear error', e);
      }
    };

    if (Platform.OS === 'web') {
      if (window.confirm('Clear ALL scan history? This cannot be undone.')) doClear();
    } else {
      Alert.alert('Clear History', 'Delete all scan entries? This cannot be undone.', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Clear All', style: 'destructive', onPress: doClear },
      ]);
    }
  };

  // ─── Filtered entries ───
  const filteredEntries = entries.filter(
    (e) => filterType === 'all' || e.analysisType === filterType || (filterType === 'single' && !e.analysisType)
  );
  const singleCount = entries.filter((e) => e.analysisType === 'single' || !e.analysisType).length;
  const compCount = entries.filter((e) => e.analysisType === 'comparison').length;
  const multiCount = entries.filter((e) => e.analysisType === 'multi_fruit').length;

  return (
    <ScrollView style={[styles.container, { backgroundColor: colors.background }]} showsVerticalScrollIndicator={false}>
      {/* ─── Page Header ─── */}
      <LinearGradient
        colors={isDark ? ['#1a1f2e', '#0f1318'] : ['#eff6ff', '#dbeafe']}
        style={styles.pageHeader}
      >
        <Animated.View entering={FadeInUp.duration(280)} style={[styles.headerContent, isDesktop && styles.headerContentDesktop]}>
          <View style={[styles.headerIcon, { backgroundColor: '#3b82f6' + '20' }]}>
            <Ionicons name="time" size={28} color="#3b82f6" />
          </View>
          <Text style={[styles.pageTitle, { color: colors.text }]}>Scan History</Text>
          <Text style={[styles.pageSubtitle, { color: colors.textSecondary }]}>
            View your saved Talisay fruit analysis results
          </Text>
        </Animated.View>
      </LinearGradient>

      <View style={[styles.content, isDesktop && styles.contentDesktop]}>
        {/* ─── Not Logged In ─── */}
        {!isAuthenticated ? (
          <Animated.View entering={FadeInUp.delay(100).duration(280)} style={[styles.emptyCard, { backgroundColor: colors.card, borderColor: colors.borderLight }]}>
            <Ionicons name="log-in-outline" size={56} color={colors.textTertiary} />
            <Text style={[styles.emptyTitle, { color: colors.text }]}>Sign In to View History</Text>
            <Text style={[styles.emptyDesc, { color: colors.textSecondary }]}>
              Log in or create an account to save and view your scan history.
            </Text>
            <Pressable onPress={() => router.push('/account')} style={[styles.actionBtn, { backgroundColor: colors.primary }]}>
              <Text style={styles.actionBtnText}>Sign In</Text>
            </Pressable>
          </Animated.View>
        ) : loading ? (
          /* ─── Loading ─── */
          <View style={styles.centered}>
            <ActivityIndicator size="large" color={colors.primary} />
            <Text style={[styles.loadingText, { color: colors.textSecondary }]}>Loading history...</Text>
          </View>
        ) : entries.length === 0 ? (
          /* ─── Empty ─── */
          <Animated.View entering={FadeInUp.delay(100).duration(280)} style={[styles.emptyCard, { backgroundColor: colors.card, borderColor: colors.borderLight }]}>
            <Ionicons name="scan-outline" size={56} color={colors.textTertiary} />
            <Text style={[styles.emptyTitle, { color: colors.text }]}>No Scans Yet</Text>
            <Text style={[styles.emptyDesc, { color: colors.textSecondary }]}>
              Analyze a Talisay fruit image on the Scan page to see your history here.
            </Text>
            <Pressable onPress={() => router.push('/scan')} style={[styles.actionBtn, { backgroundColor: colors.primary }]}>
              <Text style={styles.actionBtnText}>Go to Scan</Text>
            </Pressable>
          </Animated.View>
        ) : (
          <>
            {/* ─── Filter Tabs ─── */}
            <Animated.View entering={FadeInUp.delay(100).duration(280)} style={[styles.filterRow, { backgroundColor: colors.backgroundSecondary, borderColor: colors.borderLight }]}>
              {[
                { key: 'all', label: `All (${entries.length})`, icon: 'list' },
                { key: 'single', label: `Single (${singleCount})`, icon: 'image' },
                { key: 'comparison', label: `Compare (${compCount})`, icon: 'git-compare' },
                { key: 'multi_fruit', label: `Multiple (${multiCount})`, icon: 'apps' },
              ].map((tab) => {
                const active = filterType === tab.key;
                return (
                  <Pressable
                    key={tab.key}
                    onPress={() => setFilterType(tab.key)}
                    style={[styles.filterBtn, active && { backgroundColor: colors.primary, ...Shadows.sm }]}
                  >
                    <Ionicons name={tab.icon} size={13} color={active ? '#fff' : colors.textSecondary} />
                    <Text style={[styles.filterBtnText, { color: active ? '#fff' : colors.textSecondary }]}>{tab.label}</Text>
                  </Pressable>
                );
              })}
            </Animated.View>

            {/* ─── Toolbar ─── */}
            <Animated.View entering={FadeInUp.delay(150).duration(280)} style={styles.toolbar}>
              <Text style={[styles.resultCount, { color: colors.textSecondary }]}>
                {filteredEntries.length} {filteredEntries.length === 1 ? 'entry' : 'entries'}
              </Text>
              <Pressable onPress={handleClear} style={[styles.clearAllBtn, { borderColor: '#ef444440' }]}>
                <Ionicons name="trash-outline" size={14} color="#ef4444" />
                <Text style={styles.clearAllText}>Clear All</Text>
              </Pressable>
            </Animated.View>

            {/* ─── Cards Grid ─── */}
            {filteredEntries.length === 0 ? (
              <Animated.View entering={FadeInUp.duration(280)} style={[styles.emptyCard, { backgroundColor: colors.card, borderColor: colors.borderLight }]}>
                <Ionicons name="filter-outline" size={48} color={colors.textTertiary} />
                <Text style={[styles.emptyTitle, { color: colors.text }]}>
                  No {filterType === 'single' ? 'Single Analysis' : filterType === 'comparison' ? 'Comparison' : 'Multi-Fruit'} Scans
                </Text>
                <Text style={[styles.emptyDesc, { color: colors.textSecondary }]}>
                  Try a different filter or scan a new Talisay fruit.
                </Text>
              </Animated.View>
            ) : (
              <View style={styles.cardGrid}>
                {filteredEntries.map((entry, idx) => (
                  <HistoryCard
                    key={entry.id || entry._id || idx}
                    entry={entry}
                    onPress={() => setModalEntry(entry)}
                    delay={Math.min(idx * 50, 400)}
                    colors={colors}
                    isDark={isDark}
                    isDesktop={isDesktop}
                  />
                ))}
              </View>
            )}
          </>
        )}
      </View>

      <View style={{ height: Spacing.xxxl }} />

      {/* ─── Detail Modal ─── */}
      <DetailModal
        entry={modalEntry}
        visible={!!modalEntry}
        onClose={() => setModalEntry(null)}
        onDelete={() => modalEntry && handleDelete(modalEntry.id || modalEntry._id)}
        colors={colors}
        isDark={isDark}
        allEntries={entries}
      />
    </ScrollView>
  );
}

// ─── Styles ───
const styles = StyleSheet.create({
  container: { flex: 1 },

  /* Page Header */
  pageHeader: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.xxl,
    paddingTop: Spacing.xl,
  },
  headerContent: { gap: Spacing.sm },
  headerContentDesktop: { maxWidth: LayoutConst.maxContentWidth, alignSelf: 'center', width: '100%' },
  headerIcon: { width: 56, height: 56, borderRadius: BorderRadius.xl, alignItems: 'center', justifyContent: 'center', marginBottom: Spacing.xs, ...Shadows.sm },
  pageTitle: { ...Typography.h1, letterSpacing: -0.5 },
  pageSubtitle: { ...Typography.body, maxWidth: 500, lineHeight: 22 },

  content: { padding: Spacing.lg, gap: Spacing.md },
  contentDesktop: { maxWidth: LayoutConst.maxContentWidth, alignSelf: 'center', width: '100%', paddingHorizontal: Spacing.xxl },

  /* Empty / Login Prompt */
  emptyCard: {
    alignItems: 'center', gap: Spacing.sm, padding: Spacing.xxl,
    borderRadius: BorderRadius.lg, borderWidth: 1, ...Shadows.sm,
  },
  emptyTitle: { ...Typography.h4, textAlign: 'center' },
  emptyDesc: { ...Typography.caption, textAlign: 'center', maxWidth: 320 },
  actionBtn: {
    paddingHorizontal: Spacing.lg, paddingVertical: 10,
    borderRadius: BorderRadius.md, marginTop: Spacing.sm,
    ...Platform.select({ web: { cursor: 'pointer' } }),
  },
  actionBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },

  centered: { alignItems: 'center', justifyContent: 'center', paddingVertical: 64, gap: Spacing.md },
  loadingText: { fontSize: 14, fontWeight: '500' },

  /* Filter Row */
  filterRow: {
    flexDirection: 'row', gap: 4, padding: 5,
    borderRadius: BorderRadius.xl, borderWidth: 1,
    ...Shadows.sm,
  },
  filterBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 5, paddingVertical: 10, borderRadius: BorderRadius.lg,
    ...Platform.select({ web: { cursor: 'pointer', transition: 'all 0.2s ease' } }),
  },
  filterBtnText: { fontSize: 11, fontWeight: '600' },

  /* Toolbar */
  toolbar: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  resultCount: { fontSize: 13, fontWeight: '500' },
  clearAllBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: 10, paddingVertical: 6, borderRadius: BorderRadius.sm, borderWidth: 1,
    ...Platform.select({ web: { cursor: 'pointer' } }),
  },
  clearAllText: { color: '#ef4444', fontSize: 12, fontWeight: '600' },

  /* Card Grid */
  cardGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.md, alignItems: 'stretch' },

  /* History Card */
  historyCard: {
    borderRadius: BorderRadius.xl, borderWidth: 1,
    overflow: 'hidden', ...Shadows.md,
    ...Platform.select({ web: { cursor: 'pointer', transition: 'all 0.2s ease' } }),
  },
  historyCardDesktop: { width: 220 },
  historyCardMobile: { width: '100%' },
  cardThumb: { width: '100%', height: 130 },
  cardThumbPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  cardBody: { padding: Spacing.sm, gap: 3 },
  typeBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 3,
    alignSelf: 'flex-start', paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: 4, marginBottom: 2,
  },
  typeBadgeText: { fontSize: 9, fontWeight: '700', textTransform: 'uppercase' },
  catBadge: {
    alignSelf: 'flex-start', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6,
  },
  catBadgeText: { color: '#fff', fontSize: 11, fontWeight: '700' },
  oilYieldText: { fontSize: 16, fontWeight: '700' },
  cardMinOilText: { fontSize: 11, fontWeight: '700' },
  compLabelText: { fontSize: 10, fontWeight: '500' },
  cardDate: { fontSize: 11 },
  cardTime: { fontSize: 10 },

  /* Modal */
  modalBg: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.85)',
    justifyContent: 'center', alignItems: 'center', padding: Spacing.md,
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  modalContainer: {
    width: '100%', maxWidth: 700, height: '92%',
    borderRadius: BorderRadius.xl, overflow: 'hidden', ...Shadows.lg,
  },
  modalScroll: {
    flex: 1,
  },
  modalScrollContent: {
    paddingBottom: Spacing.lg,
  },
  modalCloseBtn: {
    position: 'absolute', top: 12, right: 12, zIndex: 10,
    padding: 8, borderRadius: 20, backgroundColor: 'rgba(0,0,0,0.5)',
    ...Platform.select({ web: { cursor: 'pointer' } }),
  },
  modalHeader: {
    padding: Spacing.lg, paddingTop: Spacing.xl,
    borderBottomWidth: 1, gap: Spacing.xs,
  },
  modalTitle: { fontSize: 20, fontWeight: '700', letterSpacing: -0.5 },
  modalMetaRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm },
  modalTypeBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6,
  },
  modalTypeText: { fontSize: 11, fontWeight: '700' },
  modalDate: { fontSize: 12 },

  modalImageSection: { position: 'relative' },
  modalImage: { width: '100%', height: 280 },
  modalImagePlaceholder: { width: '100%', height: 200, alignItems: 'center', justifyContent: 'center' },
  compLabelBox: {
    position: 'absolute', bottom: 8, left: 8,
    paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6,
  },
  compLabelBoxText: { color: '#fff', fontSize: 11, fontWeight: '600' },

  modalDetails: { padding: Spacing.lg, gap: Spacing.md },

  /* Result Row */
  modalResultRow: { flexDirection: 'row', gap: Spacing.sm },
  modalResultCol: {
    flex: 1, alignItems: 'center', gap: 4,
    padding: Spacing.md, borderRadius: BorderRadius.md, borderWidth: 1,
  },
  modalResultLabel: { fontSize: 11, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.5 },
  modalResultValue: { fontSize: 26, fontWeight: '800' },
  modalResultSub: { fontSize: 11, fontWeight: '500' },
  modalCatBadge: { paddingHorizontal: 12, paddingVertical: 4, borderRadius: 8 },
  modalCatText: { color: '#fff', fontSize: 14, fontWeight: '700' },

  /* Modal Section */
  modalSection: { gap: Spacing.sm, paddingTop: Spacing.md, borderTopWidth: 1 },
  modalSectionHeader: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm },
  modalSectionTitle: { fontSize: 14, fontWeight: '700' },

  /* Dimensions */
  dimGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.sm },
  dimItem: {
    flex: 1, minWidth: '40%', alignItems: 'center', gap: 2,
    padding: Spacing.sm, borderRadius: BorderRadius.md, borderWidth: 1,
  },
  dimLabel: { fontSize: 10, fontWeight: '600', textTransform: 'uppercase' },
  dimValue: { fontSize: 18, fontWeight: '700' },
  dimUnit: { fontSize: 10, fontWeight: '500' },

  /* Features */
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm, paddingVertical: 4 },
  featureLabel: { fontSize: 13, fontWeight: '500' },
  featureValue: { fontSize: 14, fontWeight: '700' },
  spotBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    padding: 8, borderRadius: BorderRadius.sm,
  },

  /* Interpretation */
  interpText: { fontSize: 13, lineHeight: 20 },

  /* Delete */
  deleteBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 6, paddingVertical: 12, borderRadius: BorderRadius.md, borderWidth: 1,
    marginTop: Spacing.sm,
    ...Platform.select({ web: { cursor: 'pointer' } }),
  },
  deleteBtnText: { color: '#ef4444', fontSize: 13, fontWeight: '600' },

  /* ─── Comparison Side-by-Side Panels (matches Scan page ResultDisplay) ─── */
  comparisonRow: {
    flexDirection: 'row', gap: Spacing.sm,
  },
  comparisonCol: {
    flexDirection: 'column',
  },
  dsPanel: {
    flex: 1, gap: Spacing.sm,
    borderRadius: BorderRadius.lg, borderWidth: 1, padding: Spacing.md,
    ...Shadows.sm,
  },
  dsLabelBadge: {
    alignSelf: 'flex-start', paddingHorizontal: Spacing.sm, paddingVertical: 3,
    borderRadius: BorderRadius.sm, marginBottom: 2,
  },
  dsLabelText: { fontSize: 9, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5 },
  dsImageCol: { gap: 4 },
  dsImage: { width: '100%', aspectRatio: 4 / 3, borderRadius: BorderRadius.md },
  dsFileName: { fontSize: 10, textAlign: 'center' },
  dsYieldDisplay: { alignItems: 'center', paddingVertical: Spacing.sm },
  dsYieldPercent: { fontSize: 40, fontWeight: '800', lineHeight: 46 },
  dsYieldLabel: { fontSize: 12, fontWeight: '600', marginTop: 2 },
  dsMinOilText: { fontSize: 11, fontWeight: '700', marginTop: 3 },
  dsCatBadge: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    padding: Spacing.sm, borderRadius: BorderRadius.sm,
  },
  dsCatBadgeText: { color: '#fff', fontSize: 13, fontWeight: '700' },
  dsCatBadgeConf: { color: 'rgba(255,255,255,0.9)', fontSize: 11, fontWeight: '600' },
  dsDetailsWrap: { gap: Spacing.xs },
  dsDetailCard: {
    padding: Spacing.sm, borderRadius: BorderRadius.sm, borderWidth: 1, gap: 4,
  },
  dsDetailCardHeader: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    marginBottom: 2, paddingBottom: 4, borderBottomWidth: 1, borderBottomColor: 'rgba(0,0,0,0.05)',
  },
  dsDetailIconWrap: { width: 22, height: 22, borderRadius: 4, alignItems: 'center', justifyContent: 'center' },
  dsDetailCardTitle: { fontSize: 11, fontWeight: '700' },
  dsDetailRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 2 },
  dsDetailLabel: { fontSize: 11, fontWeight: '500' },
  dsDetailValue: { fontSize: 12, fontWeight: '700' },
  oilYieldBig: { fontSize: 18, fontWeight: '800' },
  predictedValueWrap: {
    alignItems: 'flex-end',
    gap: 4,
    maxWidth: '72%',
  },
  formulaToggleBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderRadius: BorderRadius.sm,
    paddingHorizontal: 8,
    paddingVertical: 4,
    ...Platform.select({ web: { cursor: 'pointer' } }),
  },
  formulaToggleText: {
    fontSize: 11,
    fontWeight: '600',
  },
  formulaBox: {
    marginTop: 4,
    borderWidth: 1,
    borderRadius: BorderRadius.sm,
    padding: 10,
    gap: 4,
  },
  formulaTitle: {
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 2,
  },
  formulaSubTitle: {
    fontSize: 12,
    fontWeight: '700',
    marginTop: 4,
    marginBottom: 2,
  },
  formulaLine: {
    fontSize: 12,
    lineHeight: 18,
  },
  formulaEquation: {
    fontSize: 12,
    lineHeight: 18,
    fontWeight: '700',
    marginTop: 2,
  },
  dsDimGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 4 },
  dsDimCell: { flex: 1, minWidth: '40%', padding: 6, borderRadius: BorderRadius.sm, alignItems: 'center', gap: 1 },
  dsDimCellLabel: { fontSize: 9, fontWeight: '500' },
  dsDimCellValue: { fontSize: 12, fontWeight: '700' },
  dsInterpText: { fontSize: 11, lineHeight: 16 },
  compSummaryBox: {
    padding: Spacing.md, borderRadius: BorderRadius.md, borderWidth: 1, gap: Spacing.xs,
  },
});
