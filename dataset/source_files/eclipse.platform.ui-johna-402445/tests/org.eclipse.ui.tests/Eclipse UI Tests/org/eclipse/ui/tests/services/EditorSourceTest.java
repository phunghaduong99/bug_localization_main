/*******************************************************************************
 * Copyright (c) 2009 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.ui.tests.services;

import org.eclipse.core.expressions.EvaluationResult;
import org.eclipse.core.expressions.Expression;
import org.eclipse.core.expressions.ExpressionInfo;
import org.eclipse.core.expressions.IEvaluationContext;
import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.jface.util.IPropertyChangeListener;
import org.eclipse.jface.util.PropertyChangeEvent;
import org.eclipse.ui.IEditorInput;
import org.eclipse.ui.IEditorPart;
import org.eclipse.ui.IReusableEditor;
import org.eclipse.ui.ISources;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.ide.IDE;
import org.eclipse.ui.internal.util.Util;
import org.eclipse.ui.part.FileEditorInput;
import org.eclipse.ui.services.IEvaluationService;
import org.eclipse.ui.tests.api.MockReusableEditorPart;
import org.eclipse.ui.tests.harness.util.FileUtil;
import org.eclipse.ui.tests.harness.util.UITestCase;

/**
 * @since 3.5
 * 
 */
public class EditorSourceTest extends UITestCase {

	private static class MyEval implements IPropertyChangeListener {
		public int count = 0;
		public boolean currentValue;

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * org.eclipse.jface.util.IPropertyChangeListener#propertyChange(org
		 * .eclipse.jface.util.PropertyChangeEvent)
		 */
		public void propertyChange(PropertyChangeEvent event) {
			count++;
			if (event.getProperty() == IEvaluationService.RESULT
					&& event.getNewValue() instanceof Boolean) {
				currentValue = ((Boolean) event.getNewValue()).booleanValue();
			}
		}
	}

	private static class InputExpression extends Expression {
		public IEditorInput editorInput;
		public Object stateInput;

		public InputExpression(IEditorInput i) {
			editorInput = i;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * org.eclipse.core.expressions.Expression#collectExpressionInfo(org
		 * .eclipse.core.expressions.ExpressionInfo)
		 */
		public void collectExpressionInfo(ExpressionInfo info) {
			info.addVariableNameAccess(ISources.ACTIVE_EDITOR_INPUT_NAME);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * org.eclipse.core.expressions.Expression#evaluate(org.eclipse.core
		 * .expressions.IEvaluationContext)
		 */
		public EvaluationResult evaluate(IEvaluationContext context)
				throws CoreException {
			stateInput = context.getVariable(ISources.ACTIVE_EDITOR_INPUT_NAME);
			return EvaluationResult.valueOf(Util
					.equals(stateInput, editorInput));
		}

	}

	private IProject project;
	private IFile test1;
	private IFile test2;

	public EditorSourceTest(String testName) {
		super(testName);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.harness.util.UITestCase#doSetUp()
	 */
	protected void doSetUp() throws Exception {
		super.doSetUp();
		project = FileUtil.createProject("testActiveEditor");
		test1 = FileUtil.createFile("test1.mockr1", project);
		test2 = FileUtil.createFile("test2.mockr1", project);
	}

	public void testActiveEditor() throws Exception {
		IWorkbenchWindow window = openTestWindow();
		IEvaluationService es = (IEvaluationService) window
				.getService(IEvaluationService.class);
		IWorkbenchPage page = window.getActivePage();
		IEditorPart editor1 = IDE.openEditor(page, test1, true);
		assertNotNull(editor1);
		assertEquals(MockReusableEditorPart.ID1, editor1.getSite().getId());
		IEditorPart editor2 = IDE.openEditor(page, test2, true);
		assertNotNull(editor2);

		FileEditorInput input1 = new FileEditorInput(test1);
		FileEditorInput input2 = new FileEditorInput(test2);
		MyEval listener = new MyEval();
		InputExpression expr = new InputExpression(input1);
		es.addEvaluationListener(expr, listener, IEvaluationService.RESULT);
		assertFalse(listener.currentValue);
		assertEquals(1, listener.count);
		assertEquals(input2, expr.stateInput);

		IEditorPart foundEditor = page.findEditor(input1);
		assertNotNull(foundEditor);
		assertEquals(editor1, foundEditor);

		page.activate(editor1);
		processEvents();

		assertEquals(2, listener.count);
		assertEquals(input1, expr.stateInput);
		assertTrue(listener.currentValue);

		assertTrue(editor1.getClass().getName(),
				editor1 instanceof IReusableEditor);
		IReusableEditor reditor1 = (IReusableEditor) editor1;
		reditor1.setInput(input2);

		assertFalse(listener.currentValue);
		assertEquals(3, listener.count);
		assertEquals(input2, expr.stateInput);

		reditor1.setInput(input1);
		assertEquals(4, listener.count);
		assertEquals(input1, expr.stateInput);
		assertTrue(listener.currentValue);

		reditor1.setInput(input2);

		assertFalse(listener.currentValue);
		assertEquals(5, listener.count);
		assertEquals(input2, expr.stateInput);

		page.activate(editor2);
		processEvents();
		assertFalse(listener.currentValue);
		assertEquals(5, listener.count);
		assertEquals(input2, expr.stateInput);

		reditor1.setInput(input1);
		assertFalse(listener.currentValue);
		assertEquals(5, listener.count);
		assertEquals(input2, expr.stateInput);

		page.activate(editor1);
		processEvents();

		assertEquals(6, listener.count);
		assertEquals(input1, expr.stateInput);
		assertTrue(listener.currentValue);
	}
}
