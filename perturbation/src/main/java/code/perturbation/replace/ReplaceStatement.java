package code.perturbation.replace;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import code.analysis.MethodSignature;
import code.analysis.StatementAnalysis;
import code.analysis.Variables;
import code.output.result.PerturbResult;
import code.perturbation.ConstructorPerturbation;
import code.perturbation.InvocationPerturbation;
import code.perturbation.LiteralPerturbation;
import code.perturbation.OperatorPerturbation_bak;
import code.perturbation.SimilarityPerturbation;
import code.perturbation.TypePerturbation;
import code.perturbation.VariablePerturbation;
import code.perturbation.remove.FullyRemove;
import code.perturbation.utils.SelfAPRUtil;
import code.perturbation.utils.StatementType;
import spoon.reflect.code.BinaryOperatorKind;
import spoon.reflect.code.CtCodeElement;
import spoon.reflect.code.CtExpression;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.reference.CtExecutableReference;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.support.reflect.code.CtBinaryOperatorImpl;
import spoon.support.reflect.code.CtConstructorCallImpl;
import spoon.support.reflect.code.CtFieldReadImpl;
import spoon.support.reflect.code.CtIfImpl;
import spoon.support.reflect.code.CtLiteralImpl;
import spoon.support.reflect.code.CtReturnImpl;
import spoon.support.reflect.code.CtVariableReadImpl;
import spoon.support.reflect.declaration.CtConstructorImpl;
import spoon.support.reflect.reference.CtExecutableReferenceImpl;

public class ReplaceStatement {

	public static void perturb(CtElement st, StatementType type, int methStart, int methEnd, String groundTruth, int lineNo1,
			String lineNo2, String lineNo3, String lineNo4, int count) {
		String originGroundTruth = groundTruth;

		TypeFilter<CtReturnImpl> returnfilter = new TypeFilter<CtReturnImpl>(CtReturnImpl.class);
		List<CtReturnImpl> returns = st.getElements(returnfilter);
		double r = SelfAPRUtil.getRandomDouble();

		String perturbCode = null;
	

		/**
		 * operators
		 */
		if (perturbCode == null) {
			perturbCode = OperatorPerturbation_bak.perturb(st, groundTruth);
		} else if (SelfAPRUtil.getRandomDouble() > 0.7) {
			String newperturbCode = OperatorPerturbation_bak.perturb(st, perturbCode);
			if(newperturbCode!=null) {
				perturbCode = newperturbCode;
			}
		}

		/**
		 * Literal
		 */
		if (perturbCode == null) {
			perturbCode = LiteralPerturbation.perturb(st, groundTruth);
		} else if (SelfAPRUtil.getRandomDouble() > 0.7) {
			String newperturbCode = LiteralPerturbation.perturb(st, perturbCode);
			if(newperturbCode!=null) {
				perturbCode = newperturbCode;
			}
		}
		
		/**
		 * variable
		 */
		if (perturbCode == null) {
			List<String> l = VariablePerturbation.perturb(st, groundTruth, false);		
			perturbCode = l.get(SelfAPRUtil.getRandomInt(l.size()));
			} else if (SelfAPRUtil.getRandomDouble() > 0.7) {
				List<String> l = VariablePerturbation.perturb(st, perturbCode, false);	
				String newperturbCode = l.get(SelfAPRUtil.getRandomInt(l.size()));			
			if(newperturbCode!=null) {
				perturbCode = newperturbCode;
			}
		}
		
		
		/**
		 * Invocation
		 */
		if (perturbCode == null) {
			List<String> l = InvocationPerturbation.perturb(st, groundTruth);	
			String newperturbCode = l.get(SelfAPRUtil.getRandomInt(l.size()));
			
			
			
		}else if (SelfAPRUtil.getRandomDouble() > 0.7) {
			List<String> l = InvocationPerturbation.perturb(st, perturbCode);	
			String newperturbCode = l.get(SelfAPRUtil.getRandomInt(l.size()));
			if(newperturbCode!=null) {
				perturbCode = newperturbCode;
			}
		}

		
		/**
		 * similarity perturbation on the single line
		 */
		if("".equals(lineNo2)) {
			if ((groundTruth.equals(perturbCode) || perturbCode==null) && count<3) {
				System.out.println("sim statement");
				String newperturbCode = SimilarityPerturbation.perturb(st, originGroundTruth,type,0.85 - (0.5 * count),null);
				
				if(newperturbCode!=null && !"".equals(newperturbCode)) {
					perturbCode = newperturbCode;
				}
				

			}		
		}
		
		
		/**
		 * check if the perturbation fails,we iteratively perturb until the count
		 */
		if((groundTruth.equals(perturbCode) || perturbCode==null )  && count<3 ) {
			perturb( st, type, methStart, methEnd,  groundTruth, lineNo1,
					lineNo2, lineNo3,lineNo4, count+1);
		} else if(!(groundTruth.equals(perturbCode) && perturbCode!=null )){		

		HashMap<String, String> map = new HashMap<String, String>();

		map.put("lineNo1", lineNo1 + "");
		map.put("lineNo2", lineNo2+"");
		map.put("lineNo3", lineNo3+"");
		map.put("lineNo4", lineNo4);
		map.put("lineNo5", "");
		map.put("line1", perturbCode);
		map.put("line2", "");
		map.put("line3", "");
		map.put("line4", "");
		map.put("line5", "");
		map.put("groundTruth", groundTruth);
		map.put("methodStart", methStart + "");
		map.put("methodEnd", methEnd + "");
		map.put("repairAction", "[REPLACE]");

		PerturbResult.getCorruptedResult(map);
		
		System.out.println("replace "+type);

		}
		


	}

}
