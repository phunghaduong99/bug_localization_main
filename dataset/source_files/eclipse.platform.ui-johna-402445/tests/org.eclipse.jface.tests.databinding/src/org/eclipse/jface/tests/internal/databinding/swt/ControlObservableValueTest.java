/*******************************************************************************
 * Copyright (c) 2006, 2009 Brad Reynolds and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     Brad Reynolds - initial API and implementation
 *     Brad Reynolds - bug 170848
 *     Matthew Hall - bug 194734
 ******************************************************************************/

package org.eclipse.jface.tests.internal.databinding.swt;

import org.eclipse.core.databinding.observable.value.IObservableValue;
import org.eclipse.jface.databinding.conformance.util.ValueChangeEventTracker;
import org.eclipse.jface.databinding.swt.ISWTObservableValue;
import org.eclipse.jface.databinding.swt.SWTObservables;
import org.eclipse.jface.databinding.swt.WidgetProperties;
import org.eclipse.jface.resource.JFaceResources;
import org.eclipse.jface.tests.databinding.AbstractDefaultRealmTestCase;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.layout.FillLayout;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Text;

/**
 * @since 3.2
 * 
 */
public class ControlObservableValueTest extends AbstractDefaultRealmTestCase {
	private Shell shell;

	protected void setUp() throws Exception {
		super.setUp();
		
		shell = new Shell();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see junit.framework.TestCase#tearDown()
	 */
	protected void tearDown() throws Exception {
		if (shell != null && !shell.isDisposed()) {
			shell.dispose();
			shell = null;
		}
	}

	public void testSetValueEnabled() throws Exception {
		ISWTObservableValue observableValue = SWTObservables
				.observeEnabled(shell);
		Boolean value = Boolean.FALSE;
		observableValue.setValue(value);
		assertFalse(shell.isEnabled());
	}

	public void testGetValueEnabled() throws Exception {
		ISWTObservableValue value = SWTObservables.observeEnabled(shell);
		shell.setEnabled(false);
		assertEquals(Boolean.FALSE, value.getValue());
	}

	public void testGetValueTypeEnabled() throws Exception {
		ISWTObservableValue value = SWTObservables.observeEnabled(shell);
		assertEquals(boolean.class, value.getValueType());
	}

	public void testSetValueVisible() throws Exception {
		ISWTObservableValue value = SWTObservables.observeVisible(shell);
		value.setValue(Boolean.FALSE);
		assertFalse(shell.isVisible());
	}

	public void testGetValueVisible() throws Exception {
		ISWTObservableValue value = SWTObservables.observeVisible(shell);
		shell.setVisible(false);
		assertEquals(Boolean.FALSE, value.getValue());
	}

	public void testGetValueTypeVisible() throws Exception {
		ISWTObservableValue value = SWTObservables.observeVisible(shell);
		assertEquals(Boolean.TYPE, value.getValueType());
	}

	public void testSetValueForeground() throws Exception {
		ISWTObservableValue value = SWTObservables.observeForeground(shell);

		Color color = shell.getDisplay().getSystemColor(SWT.COLOR_BLACK);

		value.setValue(color);
		assertEquals(color, shell.getForeground());
	}

	public void testGetValueForeground() throws Exception {
		ISWTObservableValue value = SWTObservables.observeForeground(shell);

		Color color = shell.getDisplay().getSystemColor(SWT.COLOR_BLACK);
		shell.setForeground(color);
		assertEquals(color, value.getValue());
	}

	public void testGetValueTypeForgroundColor() throws Exception {
		ISWTObservableValue value = SWTObservables.observeForeground(shell);
		assertEquals(Color.class, value.getValueType());
	}

	public void testGetValueBackground() throws Exception {
		ISWTObservableValue value = SWTObservables.observeBackground(shell);

		Color color = shell.getDisplay().getSystemColor(SWT.COLOR_BLACK);
		shell.setBackground(color);
		assertEquals(color, value.getValue());
	}

	public void testSetValueBackground() throws Exception {
		ISWTObservableValue value = SWTObservables.observeBackground(shell);

		Color color = shell.getDisplay().getSystemColor(SWT.COLOR_BLACK);

		value.setValue(color);
		assertEquals(color, shell.getBackground());
	}

	public void testGetValueTypeBackgroundColor() throws Exception {
		ISWTObservableValue value = SWTObservables.observeBackground(shell);
		assertEquals(Color.class, value.getValueType());
	}

	public void testGetValueTypeTooltip() throws Exception {
		ISWTObservableValue value = SWTObservables.observeTooltipText(shell);
		assertEquals(String.class, value.getValueType());
	}

	public void testSetValueFont() throws Exception {
		ISWTObservableValue value = SWTObservables.observeFont(shell);

		Font font = JFaceResources.getDialogFont();

		value.setValue(font);
		assertEquals(font, shell.getFont());
	}

	public void testGetValueFont() throws Exception {
		ISWTObservableValue value = SWTObservables.observeFont(shell);

		Font font = JFaceResources.getDialogFont();
		shell.setFont(font);
		assertEquals(font, value.getValue());
	}

	public void testGetValueTypeFont() throws Exception {
		ISWTObservableValue value = SWTObservables.observeFont(shell);
		assertEquals(Font.class, value.getValueType());
	}

	public void testSetValueTooltipText() throws Exception {
		ISWTObservableValue value = SWTObservables.observeTooltipText(shell);
		String text = "text";
		value.setValue(text);
		assertEquals(text, shell.getToolTipText());
	}

	public void testGetValueTooltipText() throws Exception {
		ISWTObservableValue value = SWTObservables.observeTooltipText(shell);
		String text = "text";
		shell.setToolTipText(text);
		assertEquals(text, value.getValue());
	}

	public void testGetValueTypeTooltipText() throws Exception {
		ISWTObservableValue value = SWTObservables.observeTooltipText(shell);
		assertEquals(String.class, value.getValueType());
	}

	public void testObserveFocus() {
		shell.setLayout(new FillLayout());
		Control c1 = new Text(shell, SWT.NONE);
		Control c2 = new Text(shell, SWT.NONE);
		shell.pack();
		shell.setVisible(true);

		assertTrue(c1.setFocus());

		IObservableValue value = WidgetProperties.focused().observe(c2);
		ValueChangeEventTracker tracker = ValueChangeEventTracker
				.observe(value);

		assertTrue(c2.setFocus());

		processDisplayQueue();

		assertEquals(Boolean.TRUE, value.getValue());

		assertEquals(1, tracker.count);
		assertEquals(Boolean.FALSE, tracker.event.diff.getOldValue());
		assertEquals(Boolean.TRUE, tracker.event.diff.getNewValue());
	}

	private void processDisplayQueue() {
		while (Display.getCurrent().readAndDispatch()) {
		}
	}
}
